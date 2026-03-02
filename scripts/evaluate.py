"""Main evaluation script for accented speech recognition."""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.accent_dataset import AccentDataModule
from src.models.wav2vec2 import Wav2Vec2ASRModel
from src.eval.evaluator import ASREvaluator
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main evaluation function.
    
    Args:
        config: Hydra configuration object.
    """
    # Setup logging
    setup_logging(level=config.logging.level)
    
    logger.info("Starting ASR evaluation")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    try:
        # Load model
        logger.info("Loading model")
        if hasattr(config, "model_checkpoint") and config.model_checkpoint:
            model = Wav2Vec2ASRModel.from_pretrained(config.model_checkpoint, config.model)
        else:
            model = Wav2Vec2ASRModel(config.model)
        
        # Initialize data module
        logger.info("Setting up data module")
        data_module = AccentDataModule(config.data)
        data_module.setup()
        
        # Initialize evaluator
        logger.info("Setting up evaluator")
        evaluator = ASREvaluator(config.evaluation)
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        test_loader = data_module.test_dataloader()
        test_results = evaluator.evaluate(model, test_loader, "test")
        
        # Evaluate accent robustness if multiple accents available
        accent_distribution = data_module.get_accent_distribution()
        if len(accent_distribution) > 1:
            logger.info("Evaluating accent robustness")
            
            # Create accent-specific data loaders
            accent_loaders = {}
            for accent in accent_distribution.keys():
                # Filter dataset by accent
                accent_config = config.data.copy()
                accent_config.accent.focus_accent = accent
                
                accent_data_module = AccentDataModule(accent_config)
                accent_data_module.setup()
                accent_loaders[accent] = accent_data_module.test_dataloader()
            
            # Evaluate robustness
            robustness_results = evaluator.evaluate_accent_robustness(model, accent_loaders)
            
            # Create leaderboard
            leaderboard = evaluator.create_leaderboard(robustness_results)
            logger.info(f"\n{leaderboard}")
            
            # Save leaderboard
            leaderboard_path = Path(config.paths.outputs_dir) / "leaderboard.txt"
            leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
            with open(leaderboard_path, "w") as f:
                f.write(leaderboard)
            
            logger.info(f"Leaderboard saved to {leaderboard_path}")
        
        # Evaluate confidence calibration if requested
        if config.evaluation.confidence.enabled:
            logger.info("Evaluating confidence calibration")
            calibration_results = evaluator.evaluate_confidence_calibration(model, test_loader)
            
            logger.info(f"Confidence Calibration Results:")
            logger.info(f"  ECE: {calibration_results['ece']:.3f}")
            logger.info(f"  MCE: {calibration_results['mce']:.3f}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
