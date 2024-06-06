import torch

def calculate_iou(outputs, targets, num_classes):
    # Convert predictions and targets to binary masks
    predicted_masks = (outputs!=0)
    target_masks = (targets!=0)

    # Compute intersection and union for each class
    intersection = torch.sum(predicted_masks & target_masks)  # Sum over height and width dimensions
    union = torch.sum(predicted_masks | target_masks)

    # Calculate IoU for each class
    iou = intersection.float() / (union.float() + 1e-15)  # Add epsilon to avoid division by zero

    # Average IoU across all classes
    mean_iou = torch.mean(iou)

    return mean_iou

def calculate_per_class_ious(outputs, targets, class_names, ignore=[0,]):
    per_class_ious = {}
    for cls in range(len(class_names)):
        if cls in ignore:
            continue
        intersection = torch.sum((outputs == cls) & (targets == cls))
        union = torch.sum((outputs == cls) | (targets == cls))
        iou = intersection / (union + 1e-15)  # Smoothing to avoid division by zero
        per_class_ious[class_names[cls]] = iou.item()
    return per_class_ious

def calculate_miou(outputs, targets, class_names, ignore=[0,]):
    ious = calculate_per_class_ious(outputs, targets, class_names, ignore)
    return sum(ious.values()) / len(ious)

def calculate_precision_recall(outputs, targets, num_classes):
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)
    for cls in range(num_classes):
        true_positives[cls] = torch.sum((outputs == cls) & (targets == cls))
        false_positives[cls] = torch.sum((outputs == cls) & (targets != cls))
        false_negatives[cls] = torch.sum((outputs != cls) & (targets == cls))
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    return precision, recall