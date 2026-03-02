"""Main training script for accented speech recognition."""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.accent_dataset import AccentDataModule
from src.models.wav2vec2 import Wav2Vec2ASRModel
from src.train.trainer import ASRTrainer
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        config: Hydra configuration object.
    """
    # Setup logging
    setup_logging(level=config.logging.level)
    
    logger.info("Starting ASR training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    try:
        # Initialize data module
        logger.info("Setting up data module")
        data_module = AccentDataModule(config.data)
        data_module.setup()
        
        # Initialize model
        logger.info("Initializing model")
        model = Wav2Vec2ASRModel(config.model)
        
        # Initialize trainer
        logger.info("Setting up trainer")
        trainer = ASRTrainer(config)
        trainer.setup_model(model)
        trainer.setup_optimizer()
        
        # Get data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Train model
        logger.info("Starting training")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.epochs
        )
        
        # Log training completion
        logger.info("Training completed successfully")
        
        # Save final model
        if hasattr(model, 'save_model'):
            model_path = Path(config.paths.checkpoints_dir) / "final_model"
            model.save_model(str(model_path))
        
        # Cleanup
        trainer.cleanup()
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
