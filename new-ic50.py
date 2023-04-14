import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
import pandas as pd
import numpy as np
import random

# 读取和处理数据
def process_data(csv_file):
    df = pd.read_csv(csv_file)
    smiles = df['smiles'].tolist()
    y = df['standard_value'].tolist()
    data_list = []
    for i in range(len(smiles)):
        smile = smiles[i]
        mol = Chem.MolFromSmiles(smile)
        # 将化合物转化为图数据
        edge_index, x = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        for atom in mol.GetAtoms():
            x.append(atom_feature(atom))
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_index, _ = remove_self_loops(edge_index)
        data = Data(x=x, edge_index=edge_index, y=y[i])
        data_list.append(data)
    # 将数据集随机分为训练集、验证集和测试集
    random.shuffle(data_list)
    train_data = data_list[:int(len(data_list)*0.6)]
    val_data = data_list[int(len(data_list)*0.6):int(len(data_list)*0.8)]
    test_data = data_list[int(len(data_list)*0.8):]
    return train_data, val_data, test_data

# 定义模型
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(75, 64, heads=8, dropout=0.6)
        self.conv2 = GATConv(64*8, 32, dropout=0.6)
        self.lin1 = torch.nn.Linear(32, 16)
        self.lin2 = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.view(-1, 64*8)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

# 训练模型
def train(model, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.y.view(-1,1))
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)

# 验证模型
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(out, data.y.view(-1,1))
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# 测试模型
def test(model, loader, device):
    model.eval()
    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            y_pred.append(out.cpu().detach().numpy())
            y_true.append(data.y.cpu().detach().numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    return rmse

# 主函数
if __name__ == '__main__':
    # 处理数据
    train_data, val_data, test_data = process_data('clean_data.csv')
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    # 定义模型、优化器和设备
    model = GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 训练模型
    best_val_rmse = 1000
    for epoch in range(1, 201):
        train_loss = train(model, optimizer, train_loader, device)
        val_loss = validate(model, val_loader, device)
        val_rmse = test(model, val_loader, device)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), 'best_model.pth')
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}')

    # 测试模型
    test_rmse = test(model, test_loader, device)
    print(f'Test RMSE: {test_rmse:.4f}')

