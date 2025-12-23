import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy import stats
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)
torch.manual_seed(42)

def download_weather_data():
    """Download real weather dataset from public repository. Returns (dataframe, success_flag)."""
    import pandas as pd
    import urllib.request
    
    dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    dataset_file = "weather_data.csv"
    
    try:
        print("  Attempting to download real weather dataset...")
        urllib.request.urlretrieve(dataset_url, dataset_file)
        df = pd.read_csv(dataset_file)
        print(f"  ✓ Downloaded real dataset with {len(df)} records")
        return df, True
    except Exception as e:
        print(f"  ⚠ Could not download dataset: {e}")
        print("  Using realistic synthetic data with complex noise patterns...")
        return None, False

def generate_weather_data(n_samples=2000, anomaly_ratio=0.15):
    """Main data generation function. Tries real data first, falls back to synthetic if unavailable."""
    df, is_real = download_weather_data()
    
    if is_real and df is not None:
        return process_real_weather_data(df, n_samples, anomaly_ratio)
    
    # Fallback: Generate synthetic data if real data download fails
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    base_temp = np.random.normal(18, 8, n_normal)
    poisson_noise_temp = np.random.poisson(3, n_normal) - 3
    normal_temp = base_temp + poisson_noise_temp
    
    normal_humidity = np.random.gamma(shape=8, scale=8, size=n_normal)
    normal_humidity = np.clip(normal_humidity, 20, 95)
    
    normal_pressure = np.zeros(n_normal)
    normal_pressure[0] = np.random.normal(1013, 8)
    for i in range(1, n_normal):
        normal_pressure[i] = 0.85 * normal_pressure[i-1] + np.random.normal(0, 3)
    normal_pressure += 1013
    
    normal_wind = np.random.weibull(1.5, n_normal) * 12
    normal_wind = np.clip(normal_wind, 0, 40)
    
    rain_events = np.random.poisson(0.3, n_normal)
    normal_precip = rain_events * np.random.exponential(2, n_normal)
    
    cold_snaps = np.random.normal(-15, 8, n_anomalies//3)
    heat_waves = np.random.normal(42, 6, n_anomalies//3)
    temp_spikes = np.random.choice([-20, 48], n_anomalies - 2*(n_anomalies//3))
    anomaly_temp = np.concatenate([cold_snaps, heat_waves, temp_spikes])
    
    dry_spells = np.random.gamma(shape=2, scale=3, size=n_anomalies//2)
    saturated = np.random.normal(98, 2, n_anomalies//2)
    anomaly_humidity = np.concatenate([dry_spells, saturated])
    anomaly_humidity = np.clip(anomaly_humidity, 0, 100)
    
    low_pressure = np.random.normal(965, 12, n_anomalies//2)
    high_pressure = np.random.normal(1045, 8, n_anomalies//2)
    anomaly_pressure = np.concatenate([low_pressure, high_pressure])
    
    anomaly_wind = np.random.gamma(shape=3, scale=20, size=n_anomalies)
    anomaly_wind = np.clip(anomaly_wind, 40, 120)
    
    heavy_rain = np.random.exponential(25, n_anomalies//2)
    drought = np.zeros(n_anomalies//2)
    anomaly_precip = np.concatenate([heavy_rain, drought])
    
    burst_indices = np.random.choice(n_normal, size=n_normal//20, replace=False)
    normal_temp[burst_indices] += np.random.choice([-15, 15], size=len(burst_indices))
    normal_wind[burst_indices] += np.random.exponential(20, size=len(burst_indices))
    
    temperature = np.concatenate([normal_temp, anomaly_temp])
    humidity = np.concatenate([normal_humidity, anomaly_humidity])
    pressure = np.concatenate([normal_pressure, anomaly_pressure])
    wind_speed = np.concatenate([normal_wind, anomaly_wind])
    precipitation = np.concatenate([normal_precip, anomaly_precip])
    
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
    
    X = np.column_stack([temperature, humidity, pressure, wind_speed, precipitation])
    
    correlation_noise = np.random.multivariate_normal(
        mean=[0, 0, 0, 0, 0],
        cov=[[1, -0.3, 0.2, 0.1, -0.4],
             [-0.3, 1, -0.1, 0.2, 0.3],
             [0.2, -0.1, 1, 0.15, -0.2],
             [0.1, 0.2, 0.15, 1, 0.1],
             [-0.4, 0.3, -0.2, 0.1, 1]],
        size=n_samples
    )
    X += correlation_noise * 0.5
    
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = labels[indices]
    
    return X, y

def process_real_weather_data(df, n_samples, anomaly_ratio):
    """Process real temperature data and expand to 5 features using meteorological correlations."""
    if 'Temp' in df.columns or 'Temperature' in df.columns:
        temp_col = 'Temp' if 'Temp' in df.columns else 'Temperature'
        temps = df[temp_col].values[:n_samples]
        
        n = len(temps)
        
        humidity = 80 - temps * 1.5
        humidity = np.clip(humidity, 15, 98)
        
        pressure = np.full(n, 1013.0) + np.random.normal(0, 5, n)
        
        wind_speed = 10 + np.abs(temps - temps.mean()) * 0.3
        wind_speed = np.clip(wind_speed, 0, 50)
        
        precipitation = np.maximum(0, (20 - temps) * 0.5 + np.random.exponential(1, n))
        precipitation = np.clip(precipitation, 0, 100)
        
        X = np.column_stack([temps, humidity, pressure, wind_speed, precipitation])
        
        z_scores = np.abs(stats.zscore(X, axis=0))
        anomaly_mask = np.any(z_scores > 2.5, axis=1)
        
        n_anomalies = int(n * anomaly_ratio)
        current_anomalies = np.sum(anomaly_mask)
        
        if current_anomalies < n_anomalies:
            normal_indices = np.where(~anomaly_mask)[0]
            max_z_scores = np.max(z_scores[normal_indices], axis=1)
            extreme_indices = normal_indices[np.argsort(max_z_scores)[::-1][:n_anomalies - current_anomalies]]
            anomaly_mask[extreme_indices] = True
        elif current_anomalies > n_anomalies:
            anomaly_indices = np.where(anomaly_mask)[0]
            max_z_scores = np.max(z_scores[anomaly_indices], axis=1)
            keep_indices = anomaly_indices[np.argsort(max_z_scores)[::-1][:n_anomalies]]
            anomaly_mask = np.zeros(n, dtype=bool)
            anomaly_mask[keep_indices] = True
        
        y = anomaly_mask.astype(float)
        
        return X[:n_samples], y[:n_samples]
    
    return generate_weather_data(n_samples, anomaly_ratio)

class StandardNN(nn.Module):
    """Standard feedforward neural network without skip connections."""
    def __init__(self, input_size=5, hidden_sizes=[32, 16, 8]):
        super(StandardNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class SkipConnectionNN(nn.Module):
    """Neural network with skip connections (ResNet-style) for better gradient flow."""
    def __init__(self, input_size=5, hidden_sizes=[32, 16, 8]):
        super(SkipConnectionNN, self).__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_sizes[0])
        
        self.block1 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.skip_proj1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        
        self.block3 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.skip_proj2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        
        self.output = nn.Sequential(
            nn.Linear(hidden_sizes[2], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        identity = x
        x = self.block1(x)
        x = x + identity
        
        identity = self.skip_proj1(x)
        x = self.block2(x)
        x = x + identity
        
        identity = self.skip_proj2(x)
        x = self.block3(x)
        x = x + identity
        
        x = self.output(x)
        return x

def train_model(model, train_loader, val_loader, epochs=250, lr=0.001):
    """Train neural network and return training history (train_loss, val_loss, val_accuracy)."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                val_loss += loss.item()
                
                predicted = (outputs >= 0.5).float()
                total += y_batch.size(0)
                correct += (predicted.squeeze() == y_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate model on test set and return metrics (accuracy, precision, recall, f1, confusion_matrix)."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predicted = (outputs >= 0.5).float()
            all_predictions.extend(predicted.squeeze().numpy())
            all_labels.extend(y_batch.numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def visualize_network_architecture(standard_model, skip_model):
    """Create visual diagrams of both network architectures using torchviz."""
    try:
        from torchviz import make_dot
        import torch
        
        dummy_input = torch.randn(1, 5)
        
        print("\n  Creating network architecture visualizations...")
        
        output_standard = standard_model(dummy_input)
        dot_standard = make_dot(output_standard, params=dict(standard_model.named_parameters()))
        dot_standard.render('standard_nn_architecture', format='png', cleanup=True)
        print("  ✓ Saved: standard_nn_architecture.png")
        
        output_skip = skip_model(dummy_input)
        dot_skip = make_dot(output_skip, params=dict(skip_model.named_parameters()))
        dot_skip.render('skip_connection_nn_architecture', format='png', cleanup=True)
        print("  ✓ Saved: skip_connection_nn_architecture.png")
        
        return True
    except ImportError:
        print("\n  ⚠ torchviz not installed. Skipping architecture visualization.")
        return False
    except Exception as e:
        print(f"\n  ⚠ Could not create architecture diagrams: {e}")
        return False

def print_model_architecture(model, model_name):
    """Print model architecture and parameter counts to console."""
    print(f"\n{'='*70}")
    print(f"{model_name} ARCHITECTURE")
    print(f"{'='*70}")
    print(model)
    print(f"{'='*70}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*70}\n")

def visualize_training_comparison(standard_history, skip_history):
    """Create and save training comparison plots (loss and accuracy curves)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(standard_history[0]) + 1)
    
    axes[0].plot(epochs, standard_history[0], label='Standard NN', linewidth=2)
    axes[0].plot(epochs, skip_history[0], label='Skip Connection NN', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, standard_history[1], label='Standard NN', linewidth=2)
    axes[1].plot(epochs, skip_history[1], label='Skip Connection NN', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, standard_history[2], label='Standard NN', linewidth=2)
    axes[2].plot(epochs, skip_history[2], label='Skip Connection NN', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Validation Accuracy', fontsize=12)
    axes[2].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training comparison plot: training_comparison.png")
    plt.close()

def visualize_performance_metrics(standard_metrics, skip_metrics):
    """Create and save performance metrics comparison (bar chart and confusion matrices)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    standard_values = [
        standard_metrics['accuracy'],
        standard_metrics['precision'],
        standard_metrics['recall'],
        standard_metrics['f1']
    ]
    skip_values = [
        skip_metrics['accuracy'],
        skip_metrics['precision'],
        skip_metrics['recall'],
        skip_metrics['f1']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, standard_values, width, label='Standard NN', alpha=0.8, color='#3498db')
    axes[0].bar(x + width/2, skip_values, width, label='Skip Connection NN', alpha=0.8, color='#2ecc71')
    axes[0].set_xlabel('Metrics', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1.1])
    
    for i, (sv, skv) in enumerate(zip(standard_values, skip_values)):
        axes[0].text(i - width/2, sv + 0.02, f'{sv:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, skv + 0.02, f'{skv:.3f}', ha='center', va='bottom', fontsize=9)
    
    cm_standard = standard_metrics['confusion_matrix']
    cm_skip = skip_metrics['confusion_matrix']
    
    axes[1].text(0.25, 0.75, 'Standard NN', ha='center', fontsize=12, fontweight='bold',
                 transform=axes[1].transAxes)
    axes[1].text(0.75, 0.75, 'Skip Connection NN', ha='center', fontsize=12, fontweight='bold',
                 transform=axes[1].transAxes)
    
    cm_text_standard = f"TN: {cm_standard[0,0]}  FP: {cm_standard[0,1]}\nFN: {cm_standard[1,0]}  TP: {cm_standard[1,1]}"
    cm_text_skip = f"TN: {cm_skip[0,0]}  FP: {cm_skip[0,1]}\nFN: {cm_skip[1,0]}  TP: {cm_skip[1,1]}"
    
    axes[1].text(0.25, 0.5, cm_text_standard, ha='center', va='center', fontsize=11,
                 transform=axes[1].transAxes, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1].text(0.75, 0.5, cm_text_skip, ha='center', va='center', fontsize=11,
                 transform=axes[1].transAxes, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].axis('off')
    axes[1].set_title('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved performance metrics plot: performance_metrics.png")
    plt.close()

def save_results_summary(standard_metrics, skip_metrics, standard_history, skip_history):
    """Save numerical results summary to text file for easy reference."""
    with open('results_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("WEATHER ANOMALY DETECTION - RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("STANDARD NEURAL NETWORK RESULTS:\n")
        f.write(f"  Accuracy:  {standard_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {standard_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {standard_metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {standard_metrics['f1']:.4f}\n")
        f.write(f"  Final Training Loss: {standard_history[0][-1]:.4f}\n")
        f.write(f"  Final Validation Loss: {standard_history[1][-1]:.4f}\n\n")
        
        f.write("SKIP CONNECTION NEURAL NETWORK RESULTS:\n")
        f.write(f"  Accuracy:  {skip_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {skip_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {skip_metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {skip_metrics['f1']:.4f}\n")
        f.write(f"  Final Training Loss: {skip_history[0][-1]:.4f}\n")
        f.write(f"  Final Validation Loss: {skip_history[1][-1]:.4f}\n\n")
        
        f.write("PERFORMANCE IMPROVEMENT:\n")
        acc_improvement = (skip_metrics['accuracy'] - standard_metrics['accuracy']) * 100
        f1_improvement = (skip_metrics['f1'] - standard_metrics['f1']) * 100
        f.write(f"  Accuracy Improvement: {acc_improvement:+.2f}%\n")
        f.write(f"  F1-Score Improvement: {f1_improvement:+.2f}%\n")
        f.write("="*70 + "\n")
    
    print("✓ Saved results summary: results_summary.txt")

def main():
    """Main execution function: load data, train models, evaluate, and visualize results."""
    print("\n" + "="*70)
    print("WEATHER ANOMALY DETECTION: NEURAL NETWORK COMPARISON")
    print("="*70)
    
    print("\n[1/6] Loading/generating weather dataset...")
    X, y = generate_weather_data(n_samples=2000, anomaly_ratio=0.15)
    print(f"✓ Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Features: temperature, humidity, pressure, wind_speed, precipitation")
    print(f"  - Normal samples: {np.sum(y == 0)}")
    print(f"  - Anomaly samples: {np.sum(y == 1)}")
    
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print("\n[2/6] Initializing neural network models...")
    standard_model = StandardNN(input_size=5, hidden_sizes=[32, 16, 8])
    skip_model = SkipConnectionNN(input_size=5, hidden_sizes=[32, 16, 8])
    
    print_model_architecture(standard_model, "STANDARD NEURAL NETWORK")
    print_model_architecture(skip_model, "SKIP CONNECTION NEURAL NETWORK")
    
    visualize_network_architecture(standard_model, skip_model)
    
    print("\n[3/6] Training Standard Neural Network (250 epochs)...")
    standard_history = train_model(standard_model, train_loader, val_loader, epochs=250)
    
    print("\n[4/6] Training Skip Connection Neural Network (250 epochs)...")
    skip_history = train_model(skip_model, train_loader, val_loader, epochs=250)
    
    print("\n[5/6] Evaluating models on test set...")
    standard_metrics = evaluate_model(standard_model, test_loader)
    skip_metrics = evaluate_model(skip_model, test_loader)
    
    print("\nSTANDARD NN RESULTS:")
    print(f"  Accuracy:  {standard_metrics['accuracy']:.4f}")
    print(f"  Precision: {standard_metrics['precision']:.4f}")
    print(f"  Recall:    {standard_metrics['recall']:.4f}")
    print(f"  F1-Score:  {standard_metrics['f1']:.4f}")
    
    print("\nSKIP CONNECTION NN RESULTS:")
    print(f"  Accuracy:  {skip_metrics['accuracy']:.4f}")
    print(f"  Precision: {skip_metrics['precision']:.4f}")
    print(f"  Recall:    {skip_metrics['recall']:.4f}")
    print(f"  F1-Score:  {skip_metrics['f1']:.4f}")
    
    print("\n[6/6] Creating visualizations and saving results...")
    visualize_training_comparison(standard_history, skip_history)
    visualize_performance_metrics(standard_metrics, skip_metrics)
    save_results_summary(standard_metrics, skip_metrics, standard_history, skip_history)
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  - training_comparison.png")
    print("  - performance_metrics.png")
    print("  - results_summary.txt")
    print("  - standard_nn_architecture.png (if graphviz available)")
    print("  - skip_connection_nn_architecture.png (if graphviz available)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
