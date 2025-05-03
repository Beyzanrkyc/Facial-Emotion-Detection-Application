import torch
import torch.nn as nn
from model import *
from dataset import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from datetime import datetime
import time
import sys
import os
from tqdm import tqdm

def train_model(model, trainloader, validloader, device, epochs=100, visualize_learning_curve=True):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    valid_loss_min = np.inf
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    best_epoch = 0
    start_time = datetime.now()

    total_train_batches = len(trainloader)
    total_val_batches = len(validloader)

    for e in range(epochs):
        epoch_start = time.time()
        print(f"\n{'=' * 60}")
        print(f" EPOCH {e+1}/{epochs} ".center(60, '='))
        print(f"{'=' * 60}")

        model.train()
        running_loss = 0
        tr_accuracy = 0
       
        print(f"[1/2] Training phase:")
        train_bar = tqdm(enumerate(trainloader), total=total_train_batches,
                         desc="Training", ncols=100, leave=True)
       
        for i, (images, labels) in train_bar:
            images = images.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
           
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
           
            running_loss += loss.item()
           
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            batch_acc = torch.mean(equals.type(torch.FloatTensor)).item()
            tr_accuracy += batch_acc

            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{batch_acc:.4f}"
            })

        test_loss = 0
        accuracy = 0
        all_preds = []
        all_labels = []
       
        print(f"\n[2/2] Validation phase:")
        with torch.no_grad():
            model.eval()
            val_bar = tqdm(enumerate(validloader), total=total_val_batches,
                          desc="Validation", ncols=100, leave=True)
           
            for i, (images, labels) in val_bar:
                images = images.to(device)
                labels = labels.long().to(device)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
               
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                batch_acc = torch.mean(equals.type(torch.FloatTensor)).item()
                accuracy += batch_acc

                all_preds.extend(top_class.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_bar.set_postfix({
                    'loss': f"{criterion(log_ps, labels).item():.4f}",
                    'acc': f"{batch_acc:.4f}"
                })

        train_loss = running_loss/total_train_batches
        test_loss_val = test_loss.cpu().item()/total_val_batches
        train_acc = tr_accuracy/total_train_batches
        test_acc = accuracy/total_val_batches

        train_losses.append(train_loss)
        test_losses.append(test_loss_val)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        epoch_time = time.time() - epoch_start

        print(f"\n{'-' * 60}")
        print(f" EPOCH {e+1} SUMMARY ".center(60, '-'))
        print(f"{'-' * 60}")
        print(f"• Training Loss: {train_loss:.4f} | Training Acc: {train_acc:.4f}")
        print(f"• Val Loss:      {test_loss_val:.4f} | Val Acc:      {test_acc:.4f}")
        print(f"• Time:          {epoch_time:.2f} seconds")

        time_per_epoch = (datetime.now() - start_time).total_seconds() / (e + 1)
        est_time_remaining = time_per_epoch * (epochs - (e + 1))
        est_finish_time = datetime.now() + pd.Timedelta(seconds=est_time_remaining)
       
        print(f"• ETA:           {time.strftime('%H:%M:%S', time.gmtime(est_time_remaining))}")
        print(f"• Est. finish:   {est_finish_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'-' * 60}")

        if test_loss_val <= valid_loss_min:
            print(f"Validation loss decreased ({valid_loss_min:.6f} --> {test_loss_val:.6f})")
            print(f"Saving model from epoch {e+1}...")
            torch.save(model.state_dict(), 'best_model.pt')
            valid_loss_min = test_loss_val
            best_epoch = e + 1

            if e > 0:
                cm = confusion_matrix(all_labels, all_preds)
                save_confusion_matrix(cm, 'confusion_matrix_best.png')

    training_time = datetime.now() - start_time
    print(f"\n{'=' * 60}")
    print(f" TRAINING COMPLETE ".center(60, '='))
    print(f"{'=' * 60}")
    print(f"Total time: {training_time}")
    print(f"Best model saved at epoch {best_epoch} with validation loss: {valid_loss_min:.4f}")
    print(f"{'=' * 60}\n")

    if visualize_learning_curve:
        print("Generating learning curves...")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='Training Loss')
        plt.plot(test_losses, 'r-', label='Validation Loss')
        plt.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(test_accuracies, 'r-', label='Validation Accuracy')
        plt.axvline(x=best_epoch-1, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')

        plt.figure(figsize=(10, 6))
        plt.scatter(train_losses, train_accuracies, c='blue', alpha=0.5, label='Training')
        plt.scatter(test_losses, test_accuracies, c='red', alpha=0.5, label='Validation')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Loss')
        plt.ylabel('Accuracy')
        plt.title('Loss vs Accuracy')
        plt.legend()
        plt.savefig('loss_vs_accuracy.png', dpi=300, bbox_inches='tight')
       
        print("Learning curve visualizations saved!")
        plt.show()

    print("Evaluating best model...")
    load_and_evaluate_best_model(model, validloader, device)
   
    return model


def save_confusion_matrix(cm, filename):
    plt.figure(figsize=(10, 8))
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=emotion_classes,
               yticklabels=emotion_classes,
               cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_table_image(emotion_classes, per_class_acc, report, overall_accuracy):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.axis('tight')
    ax.axis('off')

    table_data = []
    headers = ['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support']
    table_data.append(headers)

    for i, cls in enumerate(emotion_classes):
        precision = float(report[cls]['precision']) if hasattr(report[cls]['precision'], 'item') else float(report[cls]['precision'])
        recall = float(report[cls]['recall']) if hasattr(report[cls]['recall'], 'item') else float(report[cls]['recall'])
        f1_score = float(report[cls]['f1-score']) if hasattr(report[cls]['f1-score'], 'item') else float(report[cls]['f1-score'])
        support = int(report[cls]['support']) if hasattr(report[cls]['support'], 'item') else int(report[cls]['support'])
       
        row = [
            cls,
            f"{float(per_class_acc[i]):.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1_score:.4f}",
            f"{support}"
        ]
        table_data.append(row)

    try:
        total_support = sum(int(report[cls]['support']) if hasattr(report[cls]['support'], 'item') else int(report[cls]['support']) for cls in emotion_classes)
        avg_acc = sum(float(acc) for acc in per_class_acc) / len(per_class_acc)

        macro_precision = float(report['macro avg']['precision']) if hasattr(report['macro avg']['precision'], 'item') else float(report['macro avg']['precision'])
        macro_recall = float(report['macro avg']['recall']) if hasattr(report['macro avg']['recall'], 'item') else float(report['macro avg']['recall'])
        macro_f1 = float(report['macro avg']['f1-score']) if hasattr(report['macro avg']['f1-score'], 'item') else float(report['macro avg']['f1-score'])
       
        weighted_precision = float(report['weighted avg']['precision']) if hasattr(report['weighted avg']['precision'], 'item') else float(report['weighted avg']['precision'])
        weighted_recall = float(report['weighted avg']['recall']) if hasattr(report['weighted avg']['recall'], 'item') else float(report['weighted avg']['recall'])
        weighted_f1 = float(report['weighted avg']['f1-score']) if hasattr(report['weighted avg']['f1-score'], 'item') else float(report['weighted avg']['f1-score'])

        table_data.append([
            'Macro Avg',
            f"{avg_acc:.4f}",
            f"{macro_precision:.4f}",
            f"{macro_recall:.4f}",
            f"{macro_f1:.4f}",
            f"{total_support}"
        ])

        table_data.append([
            'Weighted Avg',
            f"{avg_acc:.4f}",
            f"{weighted_precision:.4f}",
            f"{weighted_recall:.4f}",
            f"{weighted_f1:.4f}",
            f"{total_support}"
        ])

        overall_acc = float(overall_accuracy) if hasattr(overall_accuracy, 'item') else float(overall_accuracy)
        table_data.append([
            'Overall',
            f"{overall_acc:.4f}",
            '—',
            '—',
            '—',
            f"{total_support}"
        ])
    except Exception as e:
        print(f"Error creating summary rows: {e}")
        table_data.append(['Summary data unavailable due to formatting error', '', '', '', '', ''])

    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(headers)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
   
    try:
        for (i, j), cell in table.get_celld().items():
            if j == 0:  
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e6f2ff')
            if i == len(emotion_classes) or i == len(emotion_classes) + 1:  
                cell.set_facecolor('#ffffcc')
            if i == len(emotion_classes) + 2:  
                cell.set_facecolor('#e6ffe6')
    except Exception as e:
        print(f"Error formatting table cells: {e}")

    plt.suptitle('Face Emotion Recognition - Performance Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Performance summary table saved as 'performance_summary_table.png'")

def load_and_evaluate_best_model(model, dataloader, device):
    print(f"\n{'=' * 60}")
    print(f" MODEL EVALUATION ".center(60, '='))
    print(f"{'=' * 60}")
   
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
   
    print("Testing best model on validation data...")
    with torch.no_grad():
        eval_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                      desc="Evaluating", ncols=100, leave=True)
       
        for i, (images, labels) in eval_bar:
            images = images.to(device)
            labels = labels.long().to(device)

            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)

            predictions = top_class.cpu().numpy().flatten()
            true_labels = labels.cpu().numpy()

            all_preds.extend(predictions)
            all_labels.extend(true_labels)

            equals = top_class == labels.view(*top_class.shape)
            equals = equals.cpu().numpy()
           
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += equals[i]
                class_total[label] += 1

            batch_acc = equals.mean()
            eval_bar.set_postfix({'acc': f"{batch_acc:.4f}"})

    try:
            print("\nGenerating classification reports and visualizations...")
            emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

            report = classification_report(
                all_labels,
                all_preds,
                target_names=emotion_classes,
                output_dict=True,
                zero_division=0  
            )

            per_class_acc = [float(class_correct[i]/class_total[i]) if class_total[i] > 0 else 0.0 for i in range(7)]

            print("\nCreating performance summary table...")
            overall_accuracy = float(report['accuracy']) if hasattr(report['accuracy'], 'item') else float(report['accuracy'])
            create_performance_table_image(emotion_classes, per_class_acc, report, overall_accuracy)

            print("\nCreating class performance charts...")
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            bars = plt.bar(emotion_classes, per_class_acc, color='skyblue')
            plt.title('Per-Class Accuracy')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', rotation=0)

            plt.subplot(2, 1, 2)
            f1_scores = []
            for cls in emotion_classes:
                score = report[cls]['f1-score']
                if hasattr(score, 'item'):
                    score = float(score.item())
                else:
                    score = float(score)
                f1_scores.append(score)
           
            bars = plt.bar(emotion_classes, f1_scores, color='lightgreen')
            plt.title('F1 Score by Class')
            plt.ylabel('F1 Score')
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', rotation=0)
       
            plt.tight_layout()
            plt.savefig('class_performance.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("\nCreating confusion matrix visualization...")
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotion_classes,
                    yticklabels=emotion_classes,
                    cbar=False)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            plt.savefig('final_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
       
            print("All evaluation visualizations saved!")
       
    except Exception as e:
        print(f"Error in generating classification report: {e}")
        import traceback
        traceback.print_exc()

        try:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotion_classes,
                    yticklabels=emotion_classes,
                    cbar=False)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            plt.savefig('final_confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("Confusion matrix with labels saved despite errors.")
        except Exception as inner_e:
            print(f"Failed to save confusion matrix: {inner_e}")
   
    print(f"\n{'=' * 60}")
    print(f" EVALUATION COMPLETE ".center(60, '='))
    print(f"{'=' * 60}\n")

def spinner_animation(text="Loading", duration=5):
    """Display a spinner animation in the terminal."""
    spinner = "|/-\\"
    i = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        sys.stdout.write(f"\r{text} {spinner[i % len(spinner)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

    sys.stdout.write(f"\r{text} Done!      \n")
    sys.stdout.flush()

def main():
    print(f"\n{'=' * 60}")
    print(f" FACE EMOTION RECOGNITION TRAINING ".center(60, '='))
    print(f"{'=' * 60}")

    print("\nChecking for available hardware...")
    spinner_animation("Detecting hardware", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading dataset...")
    spinner_animation("Preprocessing data", 2)

    trainloader, validloader = get_dataloaders()
    print('Data processed successfully!')

    print("\nInitializing model architecture...")
    spinner_animation("Building model", 1)

    model = Face_Emotion_CNN().to(device)

    if device.type == 'cuda':
        print('GPU Found! Moving Model to CUDA.')
        if torch.cuda.get_device_name(0):
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('GPU not found! Using model with CPU.')

    print("\nPreparing for training...")
    spinner_animation("Preparing", 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training run started at: {timestamp}")

    print("\nStarting training process...")
    model = train_model(model, trainloader, validloader, device, epochs=50)

if __name__ == '__main__':
    main()