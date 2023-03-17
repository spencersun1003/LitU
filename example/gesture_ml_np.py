import torch
import numpy as np
from gesture_ml import GestureLSTMModule


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the function for LSTM forward pass
def lstm_forward(x, h, c, W_ih, W_hh, b_ih, b_hh):
    # x: input sequence of shape (seq_len, input_size)
    # h: initial hidden state of shape (num_layers, batch_size, hidden_size)
    # c: initial cell state of shape (num_layers, batch_size, hidden_size)
    # W_ih: weight matrix for input-to-hidden connections of shape (4 * hidden_size, input_size)
    # W_hh: weight matrix for hidden-to-hidden connections of shape (4 * hidden_size, hidden_size)
    # b_ih: bias vector for input-to-hidden connections of shape (4 * hidden_size,)
    # b_hh: bias vector for hidden-to-hidden connections of shape (4 * hidden_size,)
    
    # Convert input sequence to matrix form
    # x_mat = np.concatenate((x, h), axis=1)  # shape: (batch_size, input_size + hidden_size)
    
    # Compute the input-to-hidden and hidden-to-hidden transformations
    gates = np.matmul(x, np.transpose(W_ih)) + b_ih + np.matmul(h, np.transpose(W_hh)) + b_hh
    
    # Split the gates into input, forget, cell, and output gates
    i_gate, f_gate, c_gate, o_gate = np.split(gates, 4, axis=1)
    
    # Apply sigmoid activation to input, forget, and output gates
    i_gate = 1 / (1 + np.exp(-i_gate))
    f_gate = 1 / (1 + np.exp(-f_gate))
    o_gate = 1 / (1 + np.exp(-o_gate))
    
    # Apply hyperbolic tangent activation to cell gate
    c_gate = np.tanh(c_gate)
    
    # Compute the new cell state and hidden state
    c_new = f_gate * c + i_gate * c_gate
    h_new = o_gate * np.tanh(c_new)

    return h_new, c_new


# Define the function for the forward pass of the LSTM model
def lstm_model_forward(x, model):
    # x: input sequence of shape (batch_size, seq_len, input_size)
    # model: trained PyTorch LSTM model
    # Get the parameters from the trained model
    W_ih_0 = model.lstm.weight_ih_l0.detach().numpy()
    W_ih_1 = model.lstm.weight_ih_l1.detach().numpy()
    W_hh_0 = model.lstm.weight_hh_l0.detach().numpy()
    W_hh_1 = model.lstm.weight_hh_l1.detach().numpy()
    b_ih_0 = model.lstm.bias_ih_l0.detach().numpy()
    b_ih_1 = model.lstm.bias_ih_l1.detach().numpy()
    b_hh_0 = model.lstm.bias_hh_l0.detach().numpy()
    b_hh_1 = model.lstm.bias_hh_l1.detach().numpy()
    W_fc = model.fc.weight.detach().numpy()
    b_fc = model.fc.bias.detach().numpy()
    # Initialize the hidden and cell states
    num_layers = 2
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden_size = 32
    h = np.zeros((num_layers, batch_size, hidden_size))
    c = np.zeros((num_layers, batch_size, hidden_size))
    # Compute the forward pass of the LSTM model
    for t in range(seq_len):
        # Compute the forward pass of the first layer
        h[0], c[0] = lstm_forward(x[:, t, :], h[0], c[0], W_ih_0, W_hh_0, b_ih_0, b_hh_0)
        # Compute the forward pass of the second layer
        h[1], c[1] = lstm_forward(h[0], h[1], c[1], W_ih_1, W_hh_1, b_ih_1, b_hh_1)
    output = np.matmul(h[-1, :, :], np.transpose(W_fc)) + b_fc
    output = sigmoid(output)
    return output


if __name__ == '__main__':
    parameters = torch.load('../lstm_model.pt')
    model = GestureLSTMModule()
    model.load_state_dict(parameters)
    x = torch.randn(1, 20, 6)
    y = model(x)
    np_y = lstm_model_forward(x.detach().numpy(), model)
    print(y.item() - np_y.item())