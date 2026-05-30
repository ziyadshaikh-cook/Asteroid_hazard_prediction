from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    f1 = pipeline.run_pipeline()
    print(f"\nPipeline complete. Final F1: {f1}")