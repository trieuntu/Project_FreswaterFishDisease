import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torch
import torchvision
from local_utilities import LightningModel, FishDiseaseDataModule, plot_csv_logger, get_model_list
from lightning.pytorch.callbacks import ModelCheckpoint

def train_model(resnet_type, augmentation):
    model_name = f"fish-disease-{resnet_type}-{augmentation}"
    print(f"training model {model_name}")

    if augmentation == "augmented":
        augment_data = True
    else:
        augment_data = False   

    dm = FishDiseaseDataModule(height_width=(224, 224), batch_size=64, num_workers=4, augment_data=augment_data)
   
    pytorch_model = torch.hub.load('pytorch/vision', resnet_type, weights=None)
    L.pytorch.seed_everything(123)
    print(pytorch_model)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.1)

    # Set up the checkpoint callback to save the model every 10 epochs
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{model_name}/",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=-1,  # Save all checkpoints based on the interval
        save_weights_only=True,  # Save only the model weights
        every_n_epochs=2,  # Save after every 10 epochs
    )   

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices="auto",
        logger=CSVLogger(save_dir="logs/", name=model_name),
        deterministic=True,
        callbacks=[checkpoint_callback],
        log_every_n_steps=16,  # Log after every 16 batch
    )

    trainer.fit(model=lightning_model, datamodule=dm)
    torch.save(pytorch_model.state_dict(), f"{model_name}.pt") # lưu model cuối cùng sau khi chạy hết epochs
    plot_csv_logger(f"{trainer.logger.log_dir}/metrics.csv", model_name=model_name)

def test_model(resnet_type, augmentation):
    model_name = f"fish-disease-{resnet_type}-{augmentation}"
    print(f"testing model {model_name}")

    augment_data = augmentation == "augmented"
    dm = FishDiseaseDataModule(height_width=(224, 224), batch_size=64, num_workers=4, augment_data=augment_data)

    pytorch_model = torch.hub.load('pytorch/vision', resnet_type, weights=None)
    pytorch_model.load_state_dict(torch.load(f"{model_name}.pt"))  # Load the trained model weights
    pytorch_model.eval()  # Set the model to evaluation mode

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.1)

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        logger=CSVLogger(save_dir="logs/", name=f"test-{model_name}"),
    )

    results = trainer.test(model=lightning_model, datamodule=dm)
    print(f"Test results for {model_name}: {results}")
    plot_csv_logger(f"{trainer.logger.log_dir}/metrics.csv", model_name=f"test-{model_name}")


def train_models(model_list):
    for model in model_list:
        train_model(model, "baseline")
        train_model(model, "augmented")
        break # chỉ train mạng resnet18

def test_models(model_list):
    for model in model_list:
        test_model(model, "baseline")
        test_model(model, "augmented")

if __name__ == "__main__":
    model_list = get_model_list()
    train_models(model_list) 