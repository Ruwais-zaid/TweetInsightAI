import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer ,AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


'''-------------------REALTIME TWITTER SENTIMENT--------------------------------------'''
#Data Loading And Preprocessing
class TwitterDataPreprocessor:
    def __init__(self,csv_path,text_column='text',label_column='sentiment',test_size=0.2,random_state = 42):
        self.csv_path = csv_path;
        self.text_column = text_column
        self.label_column = label_column
        self.test_size = test_size
        self.random_state = random_state


        #Load data
        print(f" Loading data from { csv_path } ")
        try:
            self.df = pd.read_csv(csv_path)
            print(f" Data Loaded successfully : { len(self.df) } rows")
        except Exception as e:
            print(f' Error Loading csv_file as { e }')
            print(f' Creating sample data as fallback ')
            self.df = self._create_sample_data()
        
        print("\nDataset preview:")
        print(self.df.head())
        print("\nColumn information:")
        print(self.df.info())

        #Map string labels to integer if needed
        if self.df[self.label_column].dtype == 'object':
            self._encode_labels()
        
        #split data
        self._split_data();


    def _create_sample_data(self,n_sample = 5000):
        #create a sample text if CSV loading fails
        texts = [
            "I absolutely love this new phone! Best purchase ever!",
            "The service was terrible and the staff was rude.",
            "Just an ordinary day, nothing special happening.",
            "This movie is fantastic! I'd watch it again.",
            "I'm really disappointed with my new laptop. It's slow and crashes often."
        ]
        sentiments = [2,0,1,2,0]  # 2: positive  1 : neutral : 0 negative
        #Generate synthetic data
        sample_text  = np.random.choice(texts,n_sample)
        sample_sentiments = [ sentiments[texts.index(t)] for t in sample_text ]

        return pd.DataFrame({
            'text' : sample_text,
            'sentiment': sample_sentiments
                            })
    
    def _encode_labels(self):
        """Convert string labels to integer
        """

        label_mapping = {
            'Negative':0,
            'Positive':2,
            'Neutral': 1
        }

        if(all(label in label_mapping for label in self.df[self.label_column].unique() )):
            self.df[self.label_column] = self.df[self.label_column].map(label_mapping)
        else:
            unique_labels  = self.df[self.label_column].unique()
            label_mapping = {label: i for i,label in enumerate(unique_labels)}
            print(f'Automatically mapping labels : { label_mapping }')
            self.df[self.label_column] = self.df[self.label_column].map(label_mapping)
        self.num_classes = len(label_mapping)
        self.id2label = {v:k for k,v in label_mapping.items()}
    
    def _split_data(self):
        X = self.df[self.text_column].values
        y = self.df[self.label_column].values
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size = self.test_size,random_state=self.random_state,stratify=y)

        print(f"\nData split: {len(self.X_train)} training samples, {len(self.X_test)} testing samples")
    def clean_text(self,text):
        '''Clean and normalize tweet text'''
        #lower the text value
        text = text.lower()
        #Handle URL
        text=re.sub(r'https?://\S+', '[URL]', text)
        #Handle username
        text = re.sub(r'@\w+', '[USER]', text)
        #Handle hashtag
        text = re.sub(r'#(\w+)', r'\1', text)
        #Handle whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text;

    def prepare_data(self):
        print("Cleaning and preparing text data")
        self.X_train_cleaned = [self.clean_text(text) for text in self.X_train]
        self.X_test_cleaned = [self.clean_text(text) for text in self.X_test]

        return (self.X_train,self.X_test,self.y_train,self.y_test)
    
class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

class SentimentalClassifier(nn.Module):
    """Transformer-based sentiment classifier with attention mechanism"""
    def __init__(self, model_name="distilbert-base-uncased", num_classes=3, dropout_rate=0.3):
        super(SentimentalClassifier, self).__init__()
        
        # Load pretrained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size  # Embedding dimension
        
        # Feature extraction layer
        self.features_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),  # Gaussian Error Linear Unit for smooth activation
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism for focusing on sentiment-relevant tokens
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # Reduce dimensions
            nn.Tanh(),  # Activation function for non-linearity
            nn.Linear(hidden_size // 2, 1)  # Compute attention scores
        )
        
        # Final classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)  # Map processed features to sentiment classes
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass for sentiment classification"""
        # Pass the input text through the transformer model
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = transformer_output.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        
        # Apply attention mechanism
        attention_scores = self.attention(sequence_output)  # Shape: (batch_size, seq_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize across sequence dimension
        
        # Weighted sum of token representations
        weighted_output = torch.sum(sequence_output * attention_weights, dim=1)  # Shape: (batch_size, hidden_size)
        
        # Feature extraction and classification
        features = self.features_layers(weighted_output)
        logits = self.classifier(features)  # Extract sentiment categories
        
        return logits


class SentimentAnalyzer:
    #Handle model training , evalution, and visualization
    def __init__(self,model_name = "distilbert-base-uncased",batch_size = 32,learning_rate = 2e-5,epochs = 4, max_length =128):
        self.model_name = model_name
        self.batch_size = batch_size if isinstance(batch_size, int) else batch_size[0]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        print("Initialize tokenizer ")
        print(f" Load Tokenizer { model_name } ")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    def _prepare_loader(self,X_train,X_test,y_train,y_test):

        #Create datasets
        train_datasets = TwitterDataset(
            texts = X_train,
            labels= y_train,
            tokenizer=  self.tokenizer,
            max_length= self.max_length

        )
        test_datasets = TwitterDataset(
            texts= X_test,
            labels= y_test,
            tokenizer= self.tokenizer,
            max_length = self.max_length
        )
        #DataLoader is utility class that helps to efficiently load and process data in batches during training and evalaution
        #Batch Processing
        #Shuffeling
        #Parallelism
        #Automatic Batching

        #create data loaders
        train_loaders = DataLoader(
            train_datasets,
            batch_size = self.batch_size,
            shuffle=True

        )
        test_loaders = DataLoader(
            test_datasets,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last=True
        )
        return train_loaders,test_loaders;
    def train_model(self,train_loader,test_loader,num_classes=3):
        #Initialize Model
        model = SentimentalClassifier(model_name= self.model_name,
        num_classes = num_classes).to(device)  #model save to CPU OR GPU

        self.best_model = model
        #Initialize optimizer;
        optimizer = AdamW(
            model.parameters(),
            lr = self.learning_rate,
            weight_decay = 0.01  #help to regularize the model
        )
        #Learning rate scheduler
        total_steps = len(train_loader) * self.epochs 
        scheduler = get_linear_schedule_with_warmup( #it adjust the learning rate dynamically
            optimizer,
            num_warmup_steps=int(0.1*total_steps),#10% of training are used for gradual increases in learning rate

            num_training_steps = total_steps
        )
        #Loss Function
        criterion = nn.CrossEntropyLoss()  #its a classification problem of multiple classes
        #Training loop
        best_accuracy = 0;
        #Initialize training history
        history={
            'train_loss':[],
            'val_loss':[],
            'val_accuracy':[]
        }
        print("\n Start Training...")
        for epoch in range(self.epochs):
            print(f"\n Epoch { epoch+1 }/{self.epochs}")
            print("-"*40)
            #Training phase
            model.train()
            train_loss = 0
            max_batches = 5
            batch_counter = 0;
            for batch in tqdm(train_loader,desc='Training',leave=True, position=0):
                
                batch_counter+=1;
                if batch_counter>max_batches:
                    break
                #Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                #Forward pass 
                optimizer.zero_grad()  #Clear previous gradients
                outputs = model(input_ids,attention_mask)
                #Calculate loss
                loss =  criterion(outputs,labels)
                train_loss+=loss.item()
                #Backward pass
                loss.backward()
                #Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                #Update parameters
                optimizer.step()
                scheduler.step()
        avg_train_loss = train_loss/min(max_batches, len(train_loader))
        history['train_loss'].append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")

          #Evaluate phase
        model.eval()
        val_loss=0
        predictions=[]
        ground_truth=[]
        batch_counter_test = 0;
        max_batch_counter = 5;
        with torch.no_grad():
            for batch in tqdm(test_loader,desc="Evaluating", leave=True, position=0):
                batch_counter_test+=1
                if(batch_counter_test>max_batch_counter):
                    break
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                #Forward test Pass
                output  = model(input_ids,attention_mask)
                #Calculation test loss
                loss = criterion(output,labels)
                val_loss+=loss.item()
                #Get Predections
                _, preds = torch.max(output, dim=1)
                predictions.extend(preds.cpu().tolist())
                ground_truth.extend(labels.cpu().tolist())

        #Calculate metrics

        avg_val_loss = val_loss / min(max_batch_counter, len(test_loader))
        val_accuracy = accuracy_score(ground_truth,predictions)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")

        #Save the best model
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            self.best_model = model
            print("New Bast Model Saved")
        
        print(f"\nTraining completed. Best validation accuracy: {best_accuracy:.4f}")
        self.history = history
        return self.best_model
    def evaluate_model(self,test_loader,model=None):
        """
        Evaluate the trained model.
        
        Args:
            test_loader: Testing data loader
            model: Model to evaluate (uses best_model if None)
            
        Returns:
            predictions, ground_truth: Model predictions and true labels
        """
        if model is None:
            model = self.best_model
        model.eval()
        predictions = []
        ground_truth = []

        print("\n Evaluating model on test data....")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                output = model(input_ids,attention_mask)
                _,preds = torch.max(output,dim = 1)
                predictions.extend(preds.cpu().tolist())
                ground_truth.extend(labels.cpu().tolist())
        
        #Calculate metrics
        accuracy = accuracy_score(ground_truth,predictions)
        print(f"\n Test Accuracy : { accuracy:.4f }")
        print(f"\n Classification Report \n { classification_report(ground_truth,predictions) }")

        #save results
        self.predictions  =predictions
        self.ground_truth = ground_truth
        return predictions,ground_truth
    
    def visualize_results(self,class_names=None):
        """
        Visualize training history and confusion matrix.
        
        Args:
            class_names: Names of sentiment classes
        """
        #Default class if not provided 
        if class_names is None or class_names==[]:
            class_names = ["Negative","Positive","Neutral"]

        #set up figure
        plt.figure(figsize=(15,10))

        #Plot trainig history
        plt.subplot(2,2,1)
        plt.plot(self.history['train_loss'],label = 'Train Loss')
        plt.plot(self.history['val_loss'],label = "Validation Loss")
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        #Plot accuracy history
        plt.subplot(2,2,2)
        plt.plot(self.history['val_accuracy'],label = "Validation Accuracy")
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        #Plot confusion matrix
        plt.subplot(2,1,2)
        cm = confusion_matrix(self.ground_truth,self.predictions)
        sns.heatmap(cm,annot=True,fmt='d',cmap='orange',xticklabels=class_names,yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('sentiment_analysis_results.png')
        print("\nResults visualization saved as 'sentiment_analysis_results.png'")
        plt.show()

    def predict_sentiment(self,texts,id2label = None):
        """
        Predict sentiment for new texts.
        
        Args:
            texts: List of texts to analyze
            id2label: Dictionary mapping label IDs to label names
            
        Returns:
            predictions: Predicted sentiments
        """
        if id2label is None:
            id2label = {0:'Negative',1:'Neutral',2:'Positive'}

        #clean texts
        cleaned_texts = [re.sub(r'https?://\S+', '[URL]',text.lower()) for text in texts]
        cleaned_texts = [re.sub(r'@\w+', '[USER]', text) for text in cleaned_texts]

        #Create Datasets
        encoded_text = self.tokenizer(
            cleaned_texts,
            max_length = self.max_length,
            padding = 'max_length',
            truncation=True,
            return_tensors ='pt'

        )

        #Move to device
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        #Get Predictions
        """self.best_model.eval(): Puts the model in evaluation mode (disables dropout, batch norm updates).

          torch.no_grad(): Prevents unnecessary gradient computation to save memory and increase speed.

          torch.max(outputs, dim=1): Extracts the highest probability class index (sentiment prediction)."""
        self.best_model.eval()
        with torch.no_grad():
            outputs = self.best_model(input_ids,attention_mask)
            _,preds = torch.max(outputs,dim=1)
        
        #Conver to labels
        predictions_labels = [id2label[pred] for pred in preds.cpu().tolist()]
        #create result
        result=[]
        for text,sentiment in zip(texts,predictions_labels):
            result.append({
                'text':text,
                'sentiment':sentiment
            })
        return result



# ===========================================================
# MAIN FUNCTIONS   
# ===========================================================

def main(file_path):
    print("="*80)
    print("TWITTER SENTIMENT ANALYSIS USING TRANSFORMERS")
    print("=" * 80)

    #Configuration
    text_col = input("Enter text column name (or press Enter for default 'text'): ").strip() or "text"
    label_col = input("Enter label column name (or press Enter for default 'sentiment'): ").strip() or "sentiment"
    model_name = input("Enter model name (or press Enter for distilbert-base-uncased): ").strip() or "distilbert-base-uncased"

    #Load and preprocess the data

    preprocessor = TwitterDataPreprocessor(
        csv_path=file_path,
        text_column=text_col,
        label_column=label_col
    )
    X_train,y_train,X_test,y_test = preprocessor.prepare_data()

    #Initialize the sentimental analysis
    analyzer = SentimentAnalyzer(model_name=model_name)
    print(len(X_train))
    print(len(y_train))
    #prepare data loader
    train_loader,test_loader = analyzer._prepare_loader(X_train,y_train,X_test,y_test)
    
    #Train model
    analyzer.train_model(train_loader,test_loader,num_classes=len(np.unique(y_train)))

    # Evaluate Model
    analyzer.evaluate_model(test_loader)

    #Visualize results
    try:
        analyzer.visualize_results()
    except Exception as e:
        print(f"Error visualizing results: {e}")
    
    # Interactive Predictions
    print("\n" + "=" * 80)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("=" * 80)
    while True:
        user_input = input("\n Enter text to analyze (or 'quit' to exists): ")
        if(user_input.lower()=='quit'):
            break;
        result = analyzer.predict_sentiment([user_input],preprocessor.id2label)
        print(f"\n Text: {result[0]['text']}")
        print(f"Sentiment: {result[0]['sentiment']}")

    print("\n Thanks for using the sentiment analyzer")

def preprocess_data():
    file_path = os.path.join(os.getcwd(), 'twitter-classification', 'model', 'twitter_training.csv')
    if os.path.exists(file_path):
        columns = ['tweet_id', 'labels', 'sentiments', 'text']
        df = pd.read_csv(file_path, names=columns)
        df.drop(columns=['tweet_id'], inplace=True)
        df = df[df['sentiments'] != 'Irrelevant']  # Filter out 'Irrelevant' sentiments
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        print('Removing Null values:', df.isnull().sum())
        print("Dropping Duplicates:", df.duplicated().sum())
        print("-------------------- Information About Preprocessed Data -----------------")
        print(df.info())
        visualize_data(df)
        return df
    else:
        print("Error: File not found.")
        return None

def visualize_data(df):
    print("Distribution of Sentiments")
    df['sentiments'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 10))
    plt.title('Distribution of Sentiments')
    plt.show()

    print("Distribution of Entities")
    df['labels'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 10))
    plt.title('Distribution of Entities')
    plt.show()

    print("Distribution of Reaction Entity")
    pd.crosstab(df['labels'], df['sentiments']).plot(kind='bar', figsize=(16, 6), grid=True)
    plt.title("Distribution of Reaction Entity")
    plt.show()

    print("Distribution of Sentiment Levels")
    if 'sentiments' in df.columns:
        df['sentiments'].value_counts().plot(kind='bar', colormap='viridis', grid=True, figsize=(10, 6))
        plt.xlabel("Sentiment Category")
        plt.ylabel("Count")
        plt.title("Sentiment Distribution in Data")
        plt.show()
    else:
        print("Warning: 'sentiments' column not found in dataframe.")

if __name__ == "__main__":
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")
    
    #df = preprocess_data()
    #df.to_csv("processed_data.csv", index=False)
    new_file_path = os.path.join(os.getcwd(),'twitter-classification', 'model', 'processed_data.csv')
    if(os.path.exists(new_file_path)):
        main(new_file_path)