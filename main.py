import torch
import torch.nn as nn
from gen import get as gen_get
from transformers import TrainingArguments, Trainer
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import evaluate
import numpy as np

from PIL import Image

def colors():
    return [[255,167,0]]


def convert_logits(logits, image):
    image = image.cpu().numpy()
    image = np.squeeze(image)
    image = np.transpose(image, (1, 2, 0))
    image = 255*image

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=(256,256), #image.shape[256,256],
        #image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)

    palette = colors()
    for label, color in enumerate(palette):
        color_seg[pred_seg == label, :] = color

    img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
    img = img.astype(np.uint8)                                                      
    im = Image.fromarray(img)                                                       
    im.save('output.jpg')
    return


def compute_metrics(eval_pred):
    metric = evaluate.load('mean_iou')
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    logits_tensor = nn.functional.interpolate(
                        logits_tensor,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    # Do dice coef later
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=1,
        ignore_index=0,
        reduce_labels=False,
    )
    for key, value in metrics.items():
        if type(value) is np.ndarray:
            metrics[key] = value.tolist()
    return metrics


def fine_tune():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    t_ds, e_ds = gen_get(device)
   
    # Get Model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model.to(device)

    # Training Args
    training_args = TrainingArguments(
            output_dir="test_trainer", 
            learning_rate = 6e-5,
            num_train_epochs = 15,
            per_device_train_batch_size = 20,
            per_device_eval_batch_size = 5,
            evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=t_ds,
        eval_dataset=e_ds,
        compute_metrics=compute_metrics,
        )
    trainer.train()

    image = e_ds[0]['pixel_values'].to(device)
    image = torch.unsqueeze(image, dim=0)
    outputs = model(pixel_values = image)
    logits = outputs.logits.cpu()

    convert_logits(logits=logits, image=image) 
    return


if __name__=='__main__':
    fine_tune()



