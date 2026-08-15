import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# ==================================================
# ARGUMENTOS
# ==================================================
def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--train", default="data/raw/df_balanceado_NM_Trans.csv")
    p.add_argument("--val", default="data/raw/val_final_Trans.csv")
    p.add_argument("--test", default="data/raw/test_final_Trans.csv")
    p.add_argument("--modelo", default="models/bert_multilingual_local")
    p.add_argument("--out_model", default="models/bert_NM_Final")
    p.add_argument("--out_results", default="outputs/bert_NM_Final")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# ==================================================
# MÉTRICAS
# ==================================================
def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==================================================
# TOKENIZACIÓN
# ==================================================
def tokenizar(df, tokenizer, max_length):

    ds = Dataset.from_pandas(df)

    def tok(batch):

        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    ds = ds.map(tok, batched=True)

    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return ds

# ==================================================
# MAIN
# ==================================================
def main():

    args = parse_args()

    print("\n==============================")
    print("BERT + NM")
    print("==============================")

    print("CUDA disponible:", torch.cuda.is_available())

    # ==================================================
    # CREAR CARPETAS
    # ==================================================
    os.makedirs(args.out_model, exist_ok=True)
    os.makedirs(args.out_results, exist_ok=True)

    # ==================================================
    # CARGA DE DATOS
    # ==================================================
    train_df = pd.read_csv(args.train)
    val_df   = pd.read_csv(args.val)
    test_df  = pd.read_csv(args.test)

    # ==================================================
    # RENOMBRAR COLUMNAS
    # ==================================================
    train_df = train_df.rename(
        columns={"tweet":"text", "clase":"label"}
    )

    val_df = val_df.rename(
        columns={"tweet":"text", "clase":"label"}
    )

    test_df = test_df.rename(
        columns={"tweet":"text", "clase":"label"}
    )

    # ==================================================
    # LIMPIAR TEXTO
    # ==================================================
    train_df["text"] = train_df["text"].astype(str)
    val_df["text"]   = val_df["text"].astype(str)
    test_df["text"]  = test_df["text"].astype(str)

    # ==================================================
    # MOSTRAR DISTRIBUCIÓN
    # ==================================================
    print("\nDistribución TRAIN")
    print(train_df["label"].value_counts())

    print("\nDistribución VALIDATION")
    print(val_df["label"].value_counts())

    print("\nDistribución TEST")
    print(test_df["label"].value_counts())

    # ==================================================
    # TOKENIZER
    # ==================================================
    tokenizer = AutoTokenizer.from_pretrained(args.modelo)

    train_ds = tokenizar(
        train_df,
        tokenizer,
        args.max_length
    )

    val_ds = tokenizar(
        val_df,
        tokenizer,
        args.max_length
    )

    test_ds = tokenizar(
        test_df,
        tokenizer,
        args.max_length
    )

    # ==================================================
    # MODELO
    # ==================================================
    model = AutoModelForSequenceClassification.from_pretrained(
        args.modelo,
        num_labels=2
    )

    # ==================================================
    # TRAINING ARGUMENTS
    # ==================================================
    training_args = TrainingArguments(

        output_dir=args.out_model,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        learning_rate=args.lr,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        num_train_epochs=args.epochs,

        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        load_best_model_at_end=True,

        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=2,

        fp16=torch.cuda.is_available(),

        report_to="none",

        seed=args.seed,
    )

    # ==================================================
    # TRAINER
    # ==================================================
    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=1
            )
        ]
    )

    # ==================================================
    # ENTRENAMIENTO
    # ==================================================
    trainer.train()

    # ==================================================
    # PREDICCIÓN TEST
    # ==================================================
    pred = trainer.predict(test_ds)

    y_pred = np.argmax(pred.predictions, axis=1)

    y_true = test_df["label"].values

    # ==================================================
    # CLASSIFICATION REPORT
    # ==================================================
    report = classification_report(
        y_true,
        y_pred
    )

    print("\nClassification Report")
    print(report)

    with open(
        f"{args.out_results}/classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # ==================================================
    # MATRIZ DE CONFUSIÓN
    # ==================================================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix - BERT NM")

    plt.xlabel("Predicho")
    plt.ylabel("Real")

    plt.savefig(
        f"{args.out_results}/confusion_matrix.png"
    )

    plt.close()

    # ==================================================
    # HISTORIAL
    # ==================================================
    history = trainer.state.log_history

    train_loss = []
    eval_loss = []
    f1_scores = []
    accuracy_scores = []

    epochs_train = []
    epochs_eval = []

    for log in history:

        if "loss" in log:

            train_loss.append(log["loss"])
            epochs_train.append(log["epoch"])

        if "eval_loss" in log:

            eval_loss.append(log["eval_loss"])
            epochs_eval.append(log["epoch"])

        if "eval_f1" in log:

            f1_scores.append(log["eval_f1"])

        if "eval_accuracy" in log:

            accuracy_scores.append(log["eval_accuracy"])

    # ==================================================
    # LOSS CURVE
    # ==================================================
    plt.figure(figsize=(8,6))

    plt.plot(
        epochs_train,
        train_loss,
        label="Train Loss"
    )

    plt.plot(
        epochs_eval,
        eval_loss,
        label="Validation Loss"
    )

    plt.legend()

    plt.title("Loss Curve")

    plt.savefig(
        f"{args.out_results}/loss_curve.png"
    )

    plt.close()

    # ==================================================
    # F1 CURVE
    # ==================================================
    plt.figure(figsize=(8,6))

    plt.plot(
        epochs_eval[:len(f1_scores)],
        f1_scores
    )

    plt.title("F1 Score Curve")

    plt.savefig(
        f"{args.out_results}/f1_curve.png"
    )

    plt.close()

    # ==================================================
    # ACCURACY CURVE
    # ==================================================
    plt.figure(figsize=(8,6))

    plt.plot(
        epochs_eval[:len(accuracy_scores)],
        accuracy_scores
    )

    plt.title("Accuracy Curve")

    plt.savefig(
        f"{args.out_results}/accuracy_curve.png"
    )

    plt.close()

    # ==================================================
    # ROC CURVE
    # ==================================================
    probs = torch.softmax(
        torch.tensor(pred.predictions),
        dim=1
    )[:,1].numpy()

    fpr, tpr, _ = roc_curve(
        y_true,
        probs
    )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc:.4f}"
    )

    plt.plot([0,1],[0,1],'--')

    plt.legend()

    plt.title("ROC Curve")

    plt.savefig(
        f"{args.out_results}/roc_curve.png"
    )

    plt.close()

    # ==================================================
    # EJEMPLOS DE PREDICCIÓN
    # ==================================================
    probs_all = torch.softmax(
        torch.tensor(pred.predictions),
        dim=1
    ).numpy()

    test_texts = test_df["text"].tolist()

    correctos = []
    incorrectos = []

    for i in range(len(y_true)):

        ejemplo = {
            "texto": test_texts[i],
            "real": int(y_true[i]),
            "pred": int(y_pred[i]),
            "confianza": float(np.max(probs_all[i]))
        }

        if y_true[i] == y_pred[i]:

            correctos.append(ejemplo)

        else:

            incorrectos.append(ejemplo)

    # ==================================================
    # IMPRIMIR EJEMPLOS
    # ==================================================
    print("\n==============================")
    print("EJEMPLOS CORRECTOS")
    print("==============================")

    for i, ej in enumerate(correctos[:2], 1):

        print(f"\nEjemplo Correcto {i}")

        print(f"Texto      : {ej['texto']}")
        print(f"Etiqueta   : {ej['real']}")
        print(f"Predicción : {ej['pred']}")
        print(f"Confianza  : {ej['confianza']:.4f}")

    print("\n==============================")
    print("EJEMPLOS INCORRECTOS")
    print("==============================")

    for i, ej in enumerate(incorrectos[:2], 1):

        print(f"\nEjemplo Incorrecto {i}")

        print(f"Texto      : {ej['texto']}")
        print(f"Etiqueta   : {ej['real']}")
        print(f"Predicción : {ej['pred']}")
        print(f"Confianza  : {ej['confianza']:.4f}")

    # ==================================================
    # GUARDAR EJEMPLOS
    # ==================================================
    with open(
        f"{args.out_results}/ejemplos_prediccion.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=== EJEMPLOS CORRECTOS ===\n\n")

        for i, ej in enumerate(correctos[:2], 1):

            f.write(f"Ejemplo Correcto {i}\n")
            f.write(f"Texto      : {ej['texto']}\n")
            f.write(f"Etiqueta   : {ej['real']}\n")
            f.write(f"Predicción : {ej['pred']}\n")
            f.write(f"Confianza  : {ej['confianza']:.4f}\n\n")

        f.write("\n=== EJEMPLOS INCORRECTOS ===\n\n")

        for i, ej in enumerate(incorrectos[:2], 1):

            f.write(f"Ejemplo Incorrecto {i}\n")
            f.write(f"Texto      : {ej['texto']}\n")
            f.write(f"Etiqueta   : {ej['real']}\n")
            f.write(f"Predicción : {ej['pred']}\n")
            f.write(f"Confianza  : {ej['confianza']:.4f}\n\n")

    print("\n==============================")
    print("BERT + NM COMPLETADO")
    print("==============================")

if __name__ == "__main__":
    main()
