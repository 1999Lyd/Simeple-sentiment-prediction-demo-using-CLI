import click
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
import torch
from keras.preprocessing.sequence import pad_sequences

device = torch.device("cpu")
@click.command()
@click.option('--text', prompt = 'please input your text', help='text to be predicted')

def predict(text):
    #     labeled_df = pd.read_parquet("full_raw_data.parquet.gzip")
    #     labeled_df['sentiment'] = labeled_df['sentiment'].map({"neutral":1,"positive":2,"negative":0})
    #     train_df ,test_df = train_test_split(labeled_df,test_size=0.2)
    #     train_iter = [(label,text) for label,text in zip(train_df['sentiment'].to_list(),train_df['text'].to_list())]

    #     # Build vocabulary from tokens of training set
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    tokens = tokenizer.tokenize(text)
    input_id = [tokenizer.convert_tokens_to_ids(tokens)]
    MAX_LEN = 128
    input_id = pad_sequences(input_id, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_id_t = torch.tensor(input_id).to(device)
    # Model class must be defined somewhere

    model = "models/pretrained1.pt"
    print("Loading:", model)

    net = torch.load(model,map_location=torch.device('cpu'))

    net = net.to(device)
    # https://pytorch.org/docs/stable/torchvision/models.html
    attention_masks = []
    for seq in input_id:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    mask = torch.tensor(attention_masks).to(device)




    out = net(input_id_t, token_type_ids=None, attention_mask=mask)

    classes = ["negative", "neutral", "positive"]

    prob = torch.nn.functional.softmax(out[0], dim=1)[0] * 100

    _, indices = torch.sort(out[0], descending=True)
    print([(classes[idx], prob[idx].item()) for idx in indices[0][:3]])
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:3]]

if __name__ == '__main__':
    predict()
