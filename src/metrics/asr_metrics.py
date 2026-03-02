"""Evaluation metrics for accented speech recognition."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from jiwer import compute_measures, wer, cer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ASRMetrics:
    """Metrics calculator for ASR evaluation."""
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.predictions = []
        self.references = []
        self.accents = []
        self.speaker_ids = []
        self.confidences = []
    
    def add_batch(
        self,
        predictions: List[str],
        references: List[str],
        accents: Optional[List[str]] = None,
        speaker_ids: Optional[List[str]] = None,
        confidences: Optional[List[float]] = None
    ) -> None:
        """
        Add a batch of predictions and references.
        
        Args:
            predictions: Predicted transcriptions.
            references: Reference transcriptions.
            accents: Optional accent labels.
            speaker_ids: Optional speaker IDs.
            confidences: Optional confidence scores.
        """
        self.predictions.extend(predictions)
        self.references.extend(references)
        
        if accents:
            self.accents.extend(accents)
        if speaker_ids:
            self.speaker_ids.extend(speaker_ids)
        if confidences:
            self.confidences.extend(confidences)
    
    def compute_wer(self) -> float:
        """
        Compute Word Error Rate.
        
        Returns:
            WER as a float.
        """
        if not self.predictions or not self.references:
            return 0.0
        
        return wer(self.references, self.predictions)
    
    def compute_cer(self) -> float:
        """
        Compute Character Error Rate.
        
        Returns:
            CER as a float.
        """
        if not self.predictions or not self.references:
            return 0.0
        
        return cer(self.references, self.predictions)
    
    def compute_detailed_measures(self) -> Dict[str, float]:
        """
        Compute detailed WER measures.
        
        Returns:
            Dictionary with detailed measures.
        """
        if not self.predictions or not self.references:
            return {}
        
        measures = compute_measures(self.references, self.predictions)
        
        return {
            "wer": measures["wer"],
            "mer": measures["mer"],  # Match Error Rate
            "wil": measures["wil"],  # Word Information Lost
            "wip": measures["wip"],  # Word Information Preserved
            "hits": measures["hits"],
            "substitutions": measures["substitutions"],
            "deletions": measures["deletions"],
            "insertions": measures["insertions"]
        }
    
    def compute_accent_specific_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each accent.
        
        Returns:
            Dictionary with metrics per accent.
        """
        if not self.accents:
            return {}
        
        accent_metrics = {}
        unique_accents = set(self.accents)
        
        for accent in unique_accents:
            # Filter predictions and references for this accent
            accent_preds = [pred for pred, acc in zip(self.predictions, self.accents) if acc == accent]
            accent_refs = [ref for ref, acc in zip(self.references, self.accents) if acc == accent]
            
            if accent_preds and accent_refs:
                accent_metrics[accent] = {
                    "wer": wer(accent_refs, accent_preds),
                    "cer": cer(accent_refs, accent_preds),
                    "count": len(accent_preds)
                }
        
        return accent_metrics
    
    def compute_speaker_specific_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each speaker.
        
        Returns:
            Dictionary with metrics per speaker.
        """
        if not self.speaker_ids:
            return {}
        
        speaker_metrics = {}
        unique_speakers = set(self.speaker_ids)
        
        for speaker in unique_speakers:
            # Filter predictions and references for this speaker
            speaker_preds = [pred for pred, spk in zip(self.predictions, self.speaker_ids) if spk == speaker]
            speaker_refs = [ref for ref, spk in zip(self.references, self.speaker_ids) if spk == speaker]
            
            if speaker_preds and speaker_refs:
                speaker_metrics[speaker] = {
                    "wer": wer(speaker_refs, speaker_preds),
                    "cer": cer(speaker_refs, speaker_preds),
                    "count": len(speaker_preds)
                }
        
        return speaker_metrics
    
    def compute_confidence_calibration(
        self,
        bins: int = 10,
        method: str = "equal_width"
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Compute confidence calibration metrics.
        
        Args:
            bins: Number of bins for calibration.
            method: Binning method ("equal_width" or "equal_frequency").
            
        Returns:
            Dictionary with calibration metrics.
        """
        if not self.confidences:
            return {"ece": 0.0, "mce": 0.0, "bins": []}
        
        # Convert confidences to numpy array
        confidences = np.array(self.confidences)
        
        # Compute accuracy for each prediction
        accuracies = []
        for pred, ref in zip(self.predictions, self.references):
            # Simple word-level accuracy
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            if len(ref_words) == 0:
                accuracy = 1.0 if len(pred_words) == 0 else 0.0
            else:
                # Compute word-level accuracy
                matches = sum(1 for p, r in zip(pred_words, ref_words) if p == r)
                accuracy = matches / len(ref_words)
            
            accuracies.append(accuracy)
        
        accuracies = np.array(accuracies)
        
        # Compute Expected Calibration Error (ECE)
        if method == "equal_width":
            bin_boundaries = np.linspace(0, 1, bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        else:  # equal_frequency
            bin_boundaries = np.percentile(confidences, np.linspace(0, 100, bins + 1))
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_stats = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_stats.append({
                    "bin_range": (bin_lower, bin_upper),
                    "count": in_bin.sum(),
                    "accuracy": accuracy_in_bin,
                    "confidence": avg_confidence_in_bin,
                    "gap": avg_confidence_in_bin - accuracy_in_bin
                })
        
        # Compute Maximum Calibration Error (MCE)
        mce = max([abs(stat["gap"]) for stat in bin_stats]) if bin_stats else 0.0
        
        return {
            "ece": ece,
            "mce": mce,
            "bins": bin_stats
        }
    
    def compute_all_metrics(self) -> Dict[str, Union[float, Dict]]:
        """
        Compute all available metrics.
        
        Returns:
            Dictionary with all computed metrics.
        """
        metrics = {
            "wer": self.compute_wer(),
            "cer": self.compute_cer(),
            "detailed_measures": self.compute_detailed_measures()
        }
        
        # Add accent-specific metrics if available
        if self.accents:
            metrics["accent_specific"] = self.compute_accent_specific_metrics()
        
        # Add speaker-specific metrics if available
        if self.speaker_ids:
            metrics["speaker_specific"] = self.compute_speaker_specific_metrics()
        
        # Add confidence calibration if available
        if self.confidences:
            metrics["confidence_calibration"] = self.compute_confidence_calibration()
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot confusion matrix for accent classification (if available).
        
        Args:
            save_path: Optional path to save the plot.
            figsize: Figure size.
        """
        if not self.accents:
            logger.warning("No accent labels available for confusion matrix")
            return
        
        # This would require accent predictions, which we don't have in this demo
        # In a real implementation, you would add accent classification
        logger.info("Confusion matrix plotting not implemented in this demo")
    
    def plot_calibration_curve(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot confidence calibration curve.
        
        Args:
            save_path: Optional path to save the plot.
            figsize: Figure size.
        """
        if not self.confidences:
            logger.warning("No confidence scores available for calibration curve")
            return
        
        calibration_data = self.compute_confidence_calibration()
        
        plt.figure(figsize=figsize)
        
        # Plot calibration curve
        bin_centers = [(bin_range[0] + bin_range[1]) / 2 for bin_range in [stat["bin_range"] for stat in calibration_data["bins"]]]
        accuracies = [stat["accuracy"] for stat in calibration_data["bins"]]
        confidences = [stat["confidence"] for stat in calibration_data["bins"]]
        
        plt.plot(confidences, accuracies, 'o-', label='Model')
        plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
        
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Confidence Calibration (ECE: {calibration_data["ece"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curve saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a text report of all metrics.
        
        Returns:
            Formatted report string.
        """
        metrics = self.compute_all_metrics()
        
        report = "=== ASR EVALUATION REPORT ===\n\n"
        
        # Overall metrics
        report += "Overall Metrics:\n"
        report += f"  WER: {metrics['wer']:.2%}\n"
        report += f"  CER: {metrics['cer']:.2%}\n"
        
        # Detailed measures
        if "detailed_measures" in metrics:
            dm = metrics["detailed_measures"]
            report += "\nDetailed Measures:\n"
            report += f"  Substitutions: {dm['substitutions']}\n"
            report += f"  Deletions: {dm['deletions']}\n"
            report += f"  Insertions: {dm['insertions']}\n"
            report += f"  Hits: {dm['hits']}\n"
        
        # Accent-specific metrics
        if "accent_specific" in metrics:
            report += "\nAccent-Specific Metrics:\n"
            for accent, acc_metrics in metrics["accent_specific"].items():
                report += f"  {accent}:\n"
                report += f"    WER: {acc_metrics['wer']:.2%}\n"
                report += f"    CER: {acc_metrics['cer']:.2%}\n"
                report += f"    Count: {acc_metrics['count']}\n"
        
        # Confidence calibration
        if "confidence_calibration" in metrics:
            cc = metrics["confidence_calibration"]
            report += f"\nConfidence Calibration:\n"
            report += f"  ECE: {cc['ece']:.3f}\n"
            report += f"  MCE: {cc['mce']:.3f}\n"
        
        return report
