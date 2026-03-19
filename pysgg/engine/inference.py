# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch
from tqdm import tqdm

from pysgg.config import cfg
from pysgg.data.datasets.evaluation import evaluate
from .bbox_aug import im_detect_bbox_aug
from ..utils.comm import all_gather
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, synchronize_gather=True, timer=None, logger=None):
    """

    :param model:
    :param data_loader:
    :param device:
    :param synchronize_gather:  gather the predictions during the training,
                                rather than gathering all predictions after the training
    :param timer:
    :return:
    """
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")

    # Store relation embeddings for PCA visualization
    relation_embeddings_data = []

    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]

            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                # relation detection needs the targets
                output = model(images.to(device), targets, logger=logger)

            # Collect relation embeddings after forward pass
            if hasattr(model.roi_heads, 'relation') and hasattr(model.roi_heads.relation, 'saved_relation_embeddings'):
                embeddings = model.roi_heads.relation.saved_relation_embeddings
                if embeddings is not None:
                    # Get ground truth labels for these relations
                    for i, target in enumerate(targets):
                        gt_rels = target.get_field('relation_tuple').cpu().numpy()
                        if len(gt_rels) > 0:
                            # Each embedding corresponds to a relation in the image
                            # Store embeddings with their gt labels
                            for j, gt_rel in enumerate(gt_rels):
                                if j < len(embeddings):
                                    relation_embeddings_data.append({
                                        'embedding': embeddings[j].cpu().numpy(),
                                        'gt_label': int(gt_rel[2]) if len(gt_rel) > 2 else 0,
                                        'image_id': image_ids[i]
                                    })

            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )

    # Save relation embeddings to file if configured
    if hasattr(cfg, 'SAVE_RELATION_EMBEDDINGS') and cfg.SAVE_RELATION_EMBEDDINGS:
        if hasattr(cfg, 'RELATION_EMBEDDINGS_PATH') and cfg.RELATION_EMBEDDINGS_PATH:
            import pickle
            os.makedirs(os.path.dirname(cfg.RELATION_EMBEDDINGS_PATH), exist_ok=True)

            # Extract prototypes and sigma from DPL model if available
            prototypes_data = None
            if hasattr(model.roi_heads, 'relation') and hasattr(model.roi_heads.relation, 'predictor'):
                predictor = model.roi_heads.relation.predictor
                if hasattr(predictor, 'proto_emb') and hasattr(predictor, 'gaussian_emb'):
                    proto_emb = predictor.proto_emb.data.cpu().numpy()
                    predicate_proto = predictor.proto_emb.data
                    predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)
                    gaussian = predictor.gaussian_emb(predicate_proto_norm)
                    mu, logsigma = torch.split(gaussian, predictor.cdim, dim=1)
                    sigma = logsigma.exp().data.cpu().numpy()

                    prototypes_data = {
                        'proto_emb': proto_emb,
                        'sigma': sigma,
                        'model_info': {
                            'num_rel_cls': predictor.num_rel_cls,
                            'cdim': predictor.cdim,
                            'predictor_type': type(predictor).__name__
                        }
                    }
                    logger.info(f"Extracted prototypes: shape={proto_emb.shape}, sigma shape={sigma.shape}")
                else:
                    logger.info("Model does not have proto_emb/gaussian_emb. Not saving prototypes.")

            # Save everything together
            embeddings_output = {
                'sample_embeddings': relation_embeddings_data,
                'prototypes': prototypes_data
            }

            with open(cfg.RELATION_EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(embeddings_output, f)
            logger.info(f"Saved {len(relation_embeddings_data)} sample embeddings to {cfg.RELATION_EMBEDDINGS_PATH}")
            if prototypes_data is not None:
                logger.info(f"Also saved prototypes with {prototypes_data['proto_emb'].shape[0]} predicates")

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, synchronize_gather=True):
    if not synchronize_gather:
        all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return

    if synchronize_gather:
        predictions = predictions_per_gpu
    else:
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)

    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        
        logger = logging.getLogger("pysgg.inference")
        logger.warning(
            "WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!"
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
        logger.info(f"len(image_ids) {len(image_ids)},  image_ids[-1] + 1 {image_ids[-1] + 1}")

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        logger=None,
):
    load_prediction_from_cache = cfg.TEST.ALLOW_LOAD_FROM_CACHE and output_folder is not None and os.path.exists(
        os.path.join(output_folder, "eval_results.pytorch"))
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if logger is None:
        logger = logging.getLogger("pysgg.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    if load_prediction_from_cache:
        logging.info("load_prediction_from_cache: " + os.path.join(output_folder, "eval_results.pytorch"))
        predictions = torch.load(os.path.join(output_folder, "eval_results.pytorch"),
                                 map_location=torch.device("cpu"))['predictions']
    else:
        predictions = compute_on_dataset(model, data_loader, device,
                                         synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER,
                                         timer=inference_timer, logger=logger)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if not load_prediction_from_cache:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions,
                                                                 synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

    if not is_main_process():
        return -1.0

    # if output_folder is not None and not load_prediction_from_cache:
    #    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    return evaluate(cfg=cfg,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    logger=logger,
                    **extra_args)
