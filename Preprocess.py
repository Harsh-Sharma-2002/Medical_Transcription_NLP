

class RoBERTaEmbedder:
    def __init__(self, model_name: str = 'roberta-base', device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)

    def tokenize(self, texts: List[str], max_length: int = 256):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        )

    def generate_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        self.model.eval()
        inputs = self.tokenize(texts)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        embeddings = []
        with torch.no_grad():
            for i in range(0, input_ids.size(0), batch_size):
                batch_input_ids = input_ids[i:i+batch_size].to(self.device)
                batch_attention_mask = attention_mask[i:i+batch_size].to(self.device)

                outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)


def select_top_classes_and_oversample(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    top_k: int = 5,
    random_state: int = 42,
    return_label_encoder: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, LabelEncoder]]:
    top_classes = df[label_col].value_counts().nlargest(top_k).index.tolist()
    df_top = df[df[label_col].isin(top_classes)].copy()

    embedder = RoBERTaEmbedder()
    embeddings = embedder.generate_embeddings(df_top[text_col].tolist())

    le = LabelEncoder()
    labels_encoded = le.fit_transform(df_top[label_col].tolist())

    ros = RandomOverSampler(random_state=random_state)
    embeddings_resampled, labels_resampled = ros.fit_resample(embeddings, labels_encoded)

    if return_label_encoder:
        return embeddings_resampled, labels_resampled, le
    else:
        return embeddings_resampled, labels_resampled


def plot_class_distributions(
    df: pd.DataFrame,
    label_col: str,
    resampled_labels: Union[List[str], np.ndarray],
    top_k: int = 5
):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    sns.countplot(x=df[label_col], order=df[label_col].value_counts().index, ax=axs[0])
    axs[0].set_title("Original Class Distribution (All)")
    axs[0].tick_params(axis='x', rotation=45)

    top_classes = df[label_col].value_counts().nlargest(top_k).index
    sns.countplot(x=df[df[label_col].isin(top_classes)][label_col], ax=axs[1])
    axs[1].set_title(f"Top-{top_k} Classes Before Oversampling")
    axs[1].tick_params(axis='x', rotation=45)

    if isinstance(resampled_labels[0], (int, np.integer)):
        label_counts = pd.Series(resampled_labels).value_counts().sort_index()
        axs[2].bar(label_counts.index.astype(str), label_counts.values)
    else:
        sns.countplot(x=resampled_labels, ax=axs[2])

    axs[2].set_title(f"Top-{top_k} Classes After Oversampling")
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
