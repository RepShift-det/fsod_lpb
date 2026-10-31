# """
# Main script to train DAOD models.

# """
# import os
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
# from detectron2.engine import default_argument_parser, default_setup, launch

# from daod.config import add_config
# from daod.data.datasets import register_all_datasets
# from daod.engine.trainers import (
#     BaseTrainer,
#     BaseWQTrainer,
#     BaseMosaicTrainer,
#     DATrainer,
#     AdaptiveTeacherTrainer,
#     SourceFreeAdaptiveTeacherTrainer,
#     BaseMixupTrainer,
#     BaseMosaicWQTrainer,
#     BaseMosaicWQNewTrainer,
#     SourceFreeAdaptiveTeacherSingleTrainer,
#     SourceFreeAdaptiveTeacherTrainercl  
# )

# # To register the architectures
# from daod.modeling.meta_arch import *
# from daod.modeling.proposal_generator import *
# from daod.modeling.roi_heads import *
# from daod.engine.trainers import base

# import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
# import json

# # from rs_prompter import RSPrompter
# # from anchor_generator import AnchorGenerator

# # def adjust_features_with_prompts_and_anchors(features, prompts, anchors):
# #     # this is a placeholder implementation; adapt it to your own setting
# #     # for example, prompts and anchors can be encoded into the features
# #     adjusted_features = []
# #     for feature, prompt, anchor in zip(features, prompts, anchors):
# #         # assume the feature is a 4-D tensor (batch_size, channels, height, width)
# #         # prompts and anchors are 2-D tensors (batch_size, num_prompts_or_anchors)
# #         # simply concatenate prompts and anchors along the feature channel dimension
# #         adjusted_feature = torch.cat([feature, prompt.unsqueeze(1), anchor.unsqueeze(1)], dim=1)
# #         adjusted_features.append(adjusted_feature)
# #     return adjusted_features

# def setup(args):
#     """Setup config."""
#     cfg = get_cfg()
#     add_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(cfg, args)
#     return cfg

# def evaluate_model(cfg, trainer, weights_file):
#     """Evaluate the model on a specific weights file."""
#     print(f"Evaluating model with weights: {weights_file}")
#     DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#         weights_file, resume=False
#     )
#     results = trainer.test(cfg, trainer.model)  # call the test method through the trainer
#     print(f"Evaluation results for {weights_file}: {results}")
#     return results

# def load_confidence_scores(json_file):
#     """Load confidence scores from a JSON file and sort them."""
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#     sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)
#     return sorted_data


# def main(args):
#     """Main script."""
#     cfg = setup(args)
#     # confidence_scores = load_confidence_scores(args.json_file)
#     if cfg.TRAINER == "base":
#         Trainer = BaseTrainer
#     elif cfg.TRAINER == "base_wq":
#         Trainer = BaseWQTrainer
#     elif cfg.TRAINER == "base_mosaic":
#         Trainer = BaseMosaicTrainer
#     elif cfg.TRAINER == "base_mosaic_wq":
#         Trainer = BaseMosaicWQTrainer
#     elif cfg.TRAINER == "base_mosaic_wq_new":
#         Trainer = BaseMosaicWQNewTrainer
#     elif cfg.TRAINER == "base_mixup":
#         Trainer = BaseMixupTrainer
#     elif cfg.TRAINER == "source_free_adaptive_teacher_single":
#         Trainer = SourceFreeAdaptiveTeacherSingleTrainer
#     elif cfg.TRAINER == "da":
#         Trainer = DATrainer
#     elif cfg.TRAINER == "adaptive_teacher":
#         Trainer = AdaptiveTeacherTrainer
#     elif cfg.TRAINER == "source_free_adaptive_teacher":
#         Trainer = SourceFreeAdaptiveTeacherTrainer
#     elif cfg.TRAINER == "source_free_adaptive_teacher_cl":
#         Trainer = SourceFreeAdaptiveTeacherTrainercl
#     else:
#         raise ValueError(f"Trainer {cfg.TRAINER} not found.")

#     register_all_datasets(cfg)

#     if args.eval_only:
#         model = Trainer.build_model(cfg)
#         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#             cfg.MODEL.WEIGHTS, resume=args.resume
#         )
#         return Trainer.test(cfg, model)

#     # if args.curriculum:
#     #     trainer = SourceFreeAdaptiveTeacherTrainer(cfg)
#     #     trainer.resume_or_load(resume=args.resume)
#     #     # build the data loader with the curriculum learning strategy
#     #     curriculum_data_loader = trainer.build_train_loader_with_curriculum_learning(cfg, confidence_scores)
#     #     # use the curriculum data loader for training
#     #     trainer.train_loop(0, cfg.SOLVER.MAX_ITER)
#     # else:
#     #     trainer = SourceFreeAdaptiveTeacherTrainercl(cfg)
#     #     trainer.resume_or_load(resume=args.resume)
#     #     return trainer.train()

#     # trainer = Trainer(cfg)
#     # trainer.load_sorted_samples(args.confidence_file)

#     # trainer.resume_or_load(resume=args.resume)
#     # return trainer.train()




#     if args.eval_only:
#         # get the folder path and iterate over the weight files in it
#         weights_dir = args.weights_dir
#         weights_list = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith('.pth')]

#         if not weights_list:
#             print(f"No .pth files found in the directory: {weights_dir}")
#             return

#         # load the model
#         trainer = Trainer(cfg)

#         # run validation for each weight file
#         for weights_file in weights_list:
#             evaluate_model(cfg, trainer, weights_file)

#         return

#     if args.json_file:
#         with open(args.json_file, 'r') as f:
#             sorted_samples = json.load(f)
#         trainer = Trainer(cfg)
#         trainer.resume_or_load(resume=args.resume)
#         trainer.build_train_loader_with_curriculum_learning(sorted_samples)  # assuming this method is already implemented in the Trainer
#         return trainer.train()
#     else:
#         trainer = Trainer(cfg)
#         trainer.resume_or_load(resume=args.resume)
#         return trainer.train()
    
# def default_argument_parser_with_weights_dir():
#     parser = default_argument_parser()
#     parser.add_argument(
#         "--weights_dir", type=str, help="Directory containing .pth weight files for evaluation"
#     )

#     parser.add_argument("--json_file", default="", metavar="FILE", help="path to confidence scores file")

#     parser.add_argument(
#         "--curriculum", action="store_true", help="Use curriculum learning"
#     )
#     return parser

# if __name__ == "__main__":
#     # logger = setup_logger()
#     # args = default_argument_parser().parse_args()
#     args = default_argument_parser_with_weights_dir().parse_args()


#     print("Command line args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )













# """
# Main script to train DAOD models.

# """
# import os
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
# from detectron2.engine import default_argument_parser, default_setup, launch

# from daod.config import add_config
# from daod.data.datasets import register_all_datasets
# from daod.engine.trainers import (
#     BaseTrainer,
#     BaseWQTrainer,
#     BaseMosaicTrainer,
#     DATrainer,
#     AdaptiveTeacherTrainer,
#     SourceFreeAdaptiveTeacherTrainer,
#     BaseMixupTrainer,
#     BaseMosaicWQTrainer,
#     BaseMosaicWQNewTrainer,
#     SourceFreeAdaptiveTeacherSingleTrainer
# )

# # To register the architectures
# from daod.modeling.meta_arch import *
# from daod.modeling.proposal_generator import *
# from daod.modeling.roi_heads import *
# from daod.engine.trainers import base

# import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
# import json



# def setup(args):
#     """Setup config."""
#     cfg = get_cfg()
#     add_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(cfg, args)
#     return cfg

# def evaluate_model(cfg, trainer, weights_file):
#     """Evaluate the model on a specific weights file."""
#     print(f"Evaluating model with weights: {weights_file}")
#     DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#         weights_file, resume=False
#     )
#     results = trainer.test(cfg, trainer.model)  # call the test method through the trainer
#     print(f"Evaluation results for {weights_file}: {results}")
#     return results


# def main(args):
#     """Main script."""
#     cfg = setup(args)
#     if cfg.TRAINER == "base":
#         Trainer = BaseTrainer
#     elif cfg.TRAINER == "base_wq":
#         Trainer = BaseWQTrainer
#     elif cfg.TRAINER == "base_mosaic":
#         Trainer = BaseMosaicTrainer
#     elif cfg.TRAINER == "base_mosaic_wq":
#         Trainer = BaseMosaicWQTrainer
#     elif cfg.TRAINER == "base_mosaic_wq_new":
#         Trainer = BaseMosaicWQNewTrainer
#     elif cfg.TRAINER == "base_mixup":
#         Trainer = BaseMixupTrainer
#     elif cfg.TRAINER == "source_free_adaptive_teacher_single":
#         Trainer = SourceFreeAdaptiveTeacherSingleTrainer
#     elif cfg.TRAINER == "da":
#         Trainer = DATrainer
#     elif cfg.TRAINER == "adaptive_teacher":
#         Trainer = AdaptiveTeacherTrainer
#     elif cfg.TRAINER == "source_free_adaptive_teacher":
#         Trainer = SourceFreeAdaptiveTeacherTrainer
#     else:
#         raise ValueError(f"Trainer {cfg.TRAINER} not found.")

#     register_all_datasets(cfg)

#     # if args.eval_only:
#     #     model = Trainer.build_model(cfg)
#     #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#     #         cfg.MODEL.WEIGHTS, resume=args.resume
#     #     )
#     #     return Trainer.test(cfg, model)

#     # trainer = Trainer(cfg)

#     # trainer.resume_or_load(resume=args.resume)
#     # return trainer.train()
#     if args.eval_only:
#         # get the folder path and iterate over the weight files in it
#         weights_dir = args.weights_dir
#         weights_list = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith('.pth')]

#         if not weights_list:
#             print(f"No .pth files found in the directory: {weights_dir}")
#             return

#         # load the model
#         trainer = Trainer(cfg)

#         # run validation for each weight file
#         for weights_file in weights_list:
#             evaluate_model(cfg, trainer, weights_file)

#         return

#     if args.json_file:
#         with open(args.json_file, 'r') as f:
#             sorted_samples = json.load(f)
#         trainer = Trainer(cfg)
#         trainer.resume_or_load(resume=args.resume)
#         trainer.build_train_loader_with_curriculum_learning(sorted_samples)  # assuming this method is already implemented in the Trainer
#         return trainer.train()

#     trainer = Trainer(cfg)
#     trainer.resume_or_load(resume=args.resume)
#     return trainer.train()
    
# def default_argument_parser_with_weights_dir():
#     parser = default_argument_parser()
#     parser.add_argument(
#         "--weights_dir", type=str, help="Directory containing .pth weight files for evaluation"
#     )

#     parser.add_argument(
#         "--json-file", type=str, help="Path to json file containing sorted samples for curriculum learning"
#     )
#     return parser

# if __name__ == "__main__":
#     # logger = setup_logger()
#     # args = default_argument_parser().parse_args()
#     args = default_argument_parser_with_weights_dir().parse_args()


#     print("Command line args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )


import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from daod.config import add_config
from daod.data.datasets import register_all_datasets
from daod.engine.trainers import (
    BaseTrainer,
    BaseWQTrainer,
    BaseMosaicTrainer,
    DATrainer,
    AdaptiveTeacherTrainer,
    SourceFreeAdaptiveTeacherTrainer,
    BaseMixupTrainer,
    BaseMosaicWQTrainer,
    BaseMosaicWQNewTrainer,
    SourceFreeAdaptiveTeacherSingleTrainer
)

# To register the architectures
from daod.modeling.meta_arch import *
from daod.modeling.proposal_generator import *
from daod.modeling.roi_heads import *
from daod.engine.trainers import base

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import json


def setup(args):
    """Setup config."""
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def evaluate_model(cfg, trainer, weights_file):
    """Evaluate the model on a specific weights file."""
    print(f"Evaluating model with weights: {weights_file}")
    DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        weights_file, resume=False
    )
    results = trainer.test(cfg, trainer.model)  # call the test method through the trainer
    print(f"Evaluation results for {weights_file}: {results}")
    return results

def load_confidence_scores(json_file):
    """Load confidence scores from a JSON file and sort them."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    sorted_data = sorted(data, key=lambda x: x['score'], reverse=True)
    return sorted_data


def main(args):
    """Main script."""
    cfg = setup(args)
    # confidence_scores = load_confidence_scores(args.confidence_file)
    if cfg.TRAINER == "base":
        Trainer = BaseTrainer
    elif cfg.TRAINER == "base_wq":
        Trainer = BaseWQTrainer
    elif cfg.TRAINER == "base_mosaic":
        Trainer = BaseMosaicTrainer
    elif cfg.TRAINER == "base_mosaic_wq":
        Trainer = BaseMosaicWQTrainer
    elif cfg.TRAINER == "base_mosaic_wq_new":
        Trainer = BaseMosaicWQNewTrainer
    elif cfg.TRAINER == "base_mixup":
        Trainer = BaseMixupTrainer
    elif cfg.TRAINER == "source_free_adaptive_teacher_single":
        Trainer = SourceFreeAdaptiveTeacherSingleTrainer
    elif cfg.TRAINER == "da":
        Trainer = DATrainer
    elif cfg.TRAINER == "adaptive_teacher":
        Trainer = AdaptiveTeacherTrainer
    elif cfg.TRAINER == "source_free_adaptive_teacher":
        Trainer = SourceFreeAdaptiveTeacherTrainer
    else:
        raise ValueError(f"Trainer {cfg.TRAINER} not found.")

    register_all_datasets(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return Trainer.test(cfg, model)

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()
    # if args.eval_only:
    #     # get the folder path and iterate over the weight files in it
    #     weights_dir = args.weights_dir
    #     weights_list = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith('.pth')]

    #     if not weights_list:
    #         print(f"No .pth files found in the directory: {weights_dir}")
    #         return

    #     # load the model
    #     trainer = Trainer(cfg)

    #     # run validation for each weight file
    #     for weights_file in weights_list:
    #         evaluate_model(cfg, trainer, weights_file)

    #     return

    
    
    
    # if args.json_file:
    #     with open(args.json_file, 'r') as f:
    #         sorted_samples = json.load(f)
    #     trainer = Trainer(cfg)
    #     trainer.resume_or_load(resume=args.resume)
    #     trainer.build_train_loader_with_curriculum_learning(sorted_samples)  # assuming this method is already implemented in the Trainer
    #     return trainer.train()
    # else:
    #     trainer = Trainer(cfg)
    #     trainer.resume_or_load(resume=args.resume)
    #     return trainer.train()
    
def default_argument_parser_with_weights_dir():
    parser = default_argument_parser()
    parser.add_argument(
        "--weights_dir", type=str, help="Directory containing .pth weight files for evaluation"
    )

    parser.add_argument("--json-file", default="", metavar="FILE", help="path to confidence scores file")
    return parser

if __name__ == "__main__":
    # logger = setup_logger()
    # args = default_argument_parser().parse_args()
    args = default_argument_parser_with_weights_dir().parse_args()


    print("Command line args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )