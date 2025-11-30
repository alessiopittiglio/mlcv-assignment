import time
from pathlib import Path
import torch

from fvcore.nn import FlopCountAnalysis
from mlcv_openset_segmentation.models.model_uncertainty import UncertaintyModel
from mlcv_openset_segmentation.models.model_residual import ResidualPatternLearningModel

DEVICE = "cuda"

PHASE1_CKPT = Path("checkpoints/phase1_best_segmenter.ckpt")
PHASE2_CKPT = Path("checkpoints/phase2_best_rpl.ckpt")

INPUT_SHAPE = (1, 3, 720, 1280)
RUNS = 2


def compute_gflops(model, x):
    flops = FlopCountAnalysis(model, x)
    return flops.total() / 1e9


def measure_inference_time(model, x, runs=2):
    model.eval()

    # Warm-up
    for _ in range(20):
        with torch.no_grad():
            model(x)

    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            model(x)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)

    return sum(times) / len(times)


def main():
    segmenter = UncertaintyModel.load_from_checkpoint(PHASE1_CKPT)
    backbone = segmenter.model.eval()

    model = ResidualPatternLearningModel.load_from_checkpoint(
        checkpoint_path=PHASE2_CKPT,
        base_segmenter=backbone,
        outlier_class_idx=13,
        use_energy_entropy=True,
        score_type="energy",
        use_gaussian=True,
        optimizer_params={},
        scheduler_name=None,
        scheduler_params={},
    )

    model = model.to(DEVICE)
    dummy_input = torch.randn(*INPUT_SHAPE).to(DEVICE)

    gflops = compute_gflops(model, dummy_input)
    avg_time_ms = measure_inference_time(model, dummy_input, runs=RUNS)

    print("\nBenchmark Results")
    print("-----------------")
    print(f"Input size: {INPUT_SHAPE[2]}x{INPUT_SHAPE[3]}")
    print(f"GFLOPs: {gflops:.2f}")
    print(f"Inference Time (avg over {RUNS} runs): {avg_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
