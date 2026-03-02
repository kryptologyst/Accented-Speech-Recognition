"""Evaluation system for accented speech recognition."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.device import get_device
from src.utils.logging import setup_logging, log_evaluation_results
from src.metrics.asr_metrics import ASRMetrics

logger = logging.getLogger(__name__)


class ASREvaluator:
    """Evaluator class for ASR models."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the evaluator.
        
        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.device = get_device()
        
        # Setup logging
        self.logger = setup_logging(
            level=config.get("logging", {}).get("level", "INFO")
        )
        
        # Initialize metrics
        self.metrics = ASRMetrics()
        
        self.logger.info("ASR Evaluator initialized")
    
    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        split_name: str = "test"
    ) -> Dict[str, Union[float, Dict]]:
        """
        Evaluate the model on a dataset.
        
        Args:
            model: ASR model to evaluate.
            data_loader: Data loader for evaluation.
            split_name: Name of the dataset split.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        model.eval()
        model.to(self.device)
        
        # Reset metrics
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(data_loader)
        
        self.logger.info(f"Starting evaluation on {split_name} split")
        
        with torch.no_grad():
            progress_bar = tqdm(
                data_loader,
                desc=f"Evaluating {split_name}",
                leave=False
            )
            
            for batch in progress_bar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = model(
                    input_values=batch["input_values"],
                    labels=batch.get("labels")
                )
                
                # Compute loss if labels are available
                if "labels" in batch:
                    loss = outputs.loss
                    total_loss += loss.item()
                
                # Get predictions
                predictions = model.decode(outputs.logits)
                references = batch.get("labels", [])
                
                # Add to metrics
                self.metrics.add_batch(
                    predictions=predictions,
                    references=references,
                    accents=batch.get("accents"),
                    speaker_ids=batch.get("speaker_ids")
                )
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}" if "labels" in batch else "N/A"
                })
        
        # Compute all metrics
        eval_metrics = self.metrics.compute_all_metrics()
        
        # Add loss if available
        if "labels" in next(iter(data_loader)):
            eval_metrics["loss"] = total_loss / num_batches
        
        # Log results
        log_evaluation_results(self.logger, eval_metrics, split_name)
        
        # Generate detailed analysis if requested
        if self.config.analysis.get("per_accent", False):
            eval_metrics["accent_analysis"] = self._analyze_per_accent()
        
        if self.config.analysis.get("per_speaker", False):
            eval_metrics["speaker_analysis"] = self._analyze_per_speaker()
        
        if self.config.analysis.get("error_analysis", False):
            eval_metrics["error_analysis"] = self._analyze_errors()
        
        # Save results if requested
        if self.config.output.get("save_predictions", False):
            self._save_predictions(split_name)
        
        if self.config.output.get("generate_report", False):
            self._generate_report(eval_metrics, split_name)
        
        return eval_metrics
    
    def evaluate_accent_robustness(
        self,
        model: torch.nn.Module,
        data_loaders: Dict[str, DataLoader]
    ) -> Dict[str, Dict[str, Union[float, Dict]]]:
        """
        Evaluate model robustness across different accents.
        
        Args:
            model: ASR model to evaluate.
            data_loaders: Dictionary mapping accent names to data loaders.
            
        Returns:
            Dictionary with metrics per accent.
        """
        accent_results = {}
        
        self.logger.info("Evaluating accent robustness")
        
        for accent_name, data_loader in data_loaders.items():
            self.logger.info(f"Evaluating accent: {accent_name}")
            
            accent_metrics = self.evaluate(model, data_loader, accent_name)
            accent_results[accent_name] = accent_metrics
        
        # Compute overall robustness metrics
        robustness_metrics = self._compute_robustness_metrics(accent_results)
        accent_results["robustness"] = robustness_metrics
        
        return accent_results
    
    def evaluate_confidence_calibration(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader
    ) -> Dict[str, Union[float, List[Dict]]]:
        """
        Evaluate confidence calibration of the model.
        
        Args:
            model: ASR model to evaluate.
            data_loader: Data loader for evaluation.
            
        Returns:
            Dictionary with calibration metrics.
        """
        model.eval()
        model.to(self.device)
        
        # Reset metrics
        self.metrics.reset()
        
        self.logger.info("Evaluating confidence calibration")
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Calibration evaluation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = model(input_values=batch["input_values"])
                
                # Get predictions and confidences
                predictions = model.decode(outputs.logits)
                references = batch.get("labels", [])
                
                # Compute confidence scores (using max probability)
                confidences = torch.softmax(outputs.logits, dim=-1).max(dim=-1)[0]
                confidences = confidences.mean(dim=1).cpu().tolist()
                
                # Add to metrics
                self.metrics.add_batch(
                    predictions=predictions,
                    references=references,
                    accents=batch.get("accents"),
                    speaker_ids=batch.get("speaker_ids"),
                    confidences=confidences
                )
        
        # Compute calibration metrics
        calibration_metrics = self.metrics.compute_confidence_calibration(
            bins=self.config.confidence.bins,
            method=self.config.confidence.method
        )
        
        return calibration_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _analyze_per_accent(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance per accent."""
        return self.metrics.compute_accent_specific_metrics()
    
    def _analyze_per_speaker(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance per speaker."""
        return self.metrics.compute_speaker_specific_metrics()
    
    def _analyze_errors(self) -> Dict[str, Union[int, float]]:
        """Analyze error patterns."""
        detailed_measures = self.metrics.compute_detailed_measures()
        
        error_analysis = {
            "total_words": detailed_measures.get("hits", 0) + detailed_measures.get("substitutions", 0) + detailed_measures.get("deletions", 0),
            "substitutions": detailed_measures.get("substitutions", 0),
            "deletions": detailed_measures.get("deletions", 0),
            "insertions": detailed_measures.get("insertions", 0),
            "substitution_rate": detailed_measures.get("substitutions", 0) / max(detailed_measures.get("hits", 0) + detailed_measures.get("substitutions", 0) + detailed_measures.get("deletions", 0), 1),
            "deletion_rate": detailed_measures.get("deletions", 0) / max(detailed_measures.get("hits", 0) + detailed_measures.get("substitutions", 0) + detailed_measures.get("deletions", 0), 1),
            "insertion_rate": detailed_measures.get("insertions", 0) / max(detailed_measures.get("hits", 0) + detailed_measures.get("substitutions", 0) + detailed_measures.get("deletions", 0), 1)
        }
        
        return error_analysis
    
    def _compute_robustness_metrics(
        self,
        accent_results: Dict[str, Dict[str, Union[float, Dict]]]
    ) -> Dict[str, float]:
        """Compute robustness metrics across accents."""
        wers = [result["wer"] for result in accent_results.values() if "wer" in result]
        
        if not wers:
            return {}
        
        robustness_metrics = {
            "mean_wer": sum(wers) / len(wers),
            "std_wer": torch.tensor(wers).std().item(),
            "min_wer": min(wers),
            "max_wer": max(wers),
            "wer_range": max(wers) - min(wers),
            "wer_cv": torch.tensor(wers).std().item() / torch.tensor(wers).mean().item() if torch.tensor(wers).mean() > 0 else 0.0
        }
        
        return robustness_metrics
    
    def _save_predictions(self, split_name: str) -> None:
        """Save predictions to file."""
        # This would save predictions to a file
        # Implementation depends on specific requirements
        self.logger.info(f"Predictions saved for {split_name} split")
    
    def _generate_report(
        self,
        metrics: Dict[str, Union[float, Dict]],
        split_name: str
    ) -> None:
        """Generate evaluation report."""
        report = self.metrics.generate_report()
        
        # Save report
        report_path = Path(self.config.paths.outputs_dir) / f"evaluation_report_{split_name}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write(report)
        
        self.logger.info(f"Evaluation report saved to {report_path}")
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, Union[float, Dict]]]
    ) -> str:
        """
        Create a leaderboard from evaluation results.
        
        Args:
            results: Dictionary with evaluation results.
            
        Returns:
            Formatted leaderboard string.
        """
        leaderboard = "=== ASR LEADERBOARD ===\n\n"
        
        # Overall metrics
        if "overall" in results:
            overall = results["overall"]
            leaderboard += "Overall Performance:\n"
            leaderboard += f"  WER: {overall.get('wer', 0):.2%}\n"
            leaderboard += f"  CER: {overall.get('cer', 0):.2%}\n\n"
        
        # Per-accent metrics
        accent_metrics = {}
        for key, result in results.items():
            if key not in ["overall", "robustness"] and isinstance(result, dict):
                if "wer" in result:
                    accent_metrics[key] = result["wer"]
        
        if accent_metrics:
            leaderboard += "Per-Accent Performance:\n"
            sorted_accents = sorted(accent_metrics.items(), key=lambda x: x[1])
            
            for i, (accent, wer) in enumerate(sorted_accents, 1):
                leaderboard += f"  {i}. {accent}: {wer:.2%}\n"
            
            leaderboard += "\n"
        
        # Robustness metrics
        if "robustness" in results:
            robustness = results["robustness"]
            leaderboard += "Robustness Metrics:\n"
            leaderboard += f"  Mean WER: {robustness.get('mean_wer', 0):.2%}\n"
            leaderboard += f"  WER Std: {robustness.get('std_wer', 0):.2%}\n"
            leaderboard += f"  WER Range: {robustness.get('wer_range', 0):.2%}\n"
            leaderboard += f"  Coefficient of Variation: {robustness.get('wer_cv', 0):.3f}\n"
        
        return leaderboard
