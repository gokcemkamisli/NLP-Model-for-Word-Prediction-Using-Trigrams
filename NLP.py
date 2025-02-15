#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:52:35 2024

@author: gokcemkamisli
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42) # i used this line because of getting same result and improving reproducebility

#data loading 

with h5py.File('data2.h5', 'r') as f:
    trainx = np.array(f['trainx'])  -1 # changing the base for each data 
    traind = np.array(f['traind'])  -1 
    valx = np.array(f['valx'])      -1 
    vald = np.array(f['vald'])      -1 
    testx = np.array(f['testx'])    -1 
    testd = np.array(f['testd'])    -1 


# parameters
vocab_size = 250
D = 32
P = 256
batch_size = 200
learning_rate = 0.15
momentum_rate = 0.85
max_epochs = 50
patience = 5

# shuffle training data
perm = np.random.permutation(len(trainx))
trainx = trainx[perm]
traind = traind[perm]

# initialize parameters
rng = np.random.default_rng()
R = rng.normal(0, 0.01, size=(vocab_size, D))   
W_h = rng.normal(0, 0.01, size=(3*D, P))
b_h = np.zeros(P)
W_o = rng.normal(0, 0.01, size=(P, vocab_size))
b_o = np.zeros(vocab_size)

#momentum initialization 
v_R = np.zeros_like(R)
v_W_h = np.zeros_like(W_h)
v_b_h = np.zeros_like(b_h)
v_W_o = np.zeros_like(W_o)
v_b_o = np.zeros_like(b_o)



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    # subtract max for stability
    x = x - np.max(x, axis=1, keepdims=True)
    expx = np.exp(x)
    return expx / np.sum(expx, axis=1, keepdims=True)

def cross_entropy_loss_and_grad(logits, targets):
    
    B = logits.shape[0]
    probs = softmax(logits)
    loss = -np.mean(np.log(probs[np.arange(B), targets] + 1e-12))
    grad_logits = probs
    grad_logits[np.arange(B), targets] -= 1
    grad_logits /= B
    return loss, grad_logits


def forward_backward(x_batch, d_batch, 
                     R, W_h, b_h, W_o, b_o):
    B = x_batch.shape[0]
    
    # embeddings 
    
    emb = R[x_batch, :]  # (B, 3, D)
    emb_flat = emb.reshape(B, 3*D)  

    # Hidden layer
    h_in = emb_flat @ W_h + b_h  
    h = sigmoid(h_in)            

    # Output layer
    logits = h @ W_o + b_o      
    loss, grad_logits = cross_entropy_loss_and_grad(logits, d_batch)

    # grad wrt W_o and b_o
    grad_W_o = h.T @ grad_logits  
    grad_b_o = np.sum(grad_logits, axis=0) 

    # grad wrt h
    grad_h = grad_logits @ W_o.T
    grad_h_in = grad_h * h * (1 - h)  

    # grad wrt W_h and b_h
    grad_W_h = emb_flat.T @ grad_h_in  
    grad_b_h = np.sum(grad_h_in, axis=0) 

    # grad wrt emb_flat
    grad_emb_flat = grad_h_in @ W_h.T  
    grad_emb = grad_emb_flat.reshape(B, 3, D)  

    # grad wrt R
    grad_R = np.zeros_like(R)
    
    for i in range(B):
        for j in range(3):
            grad_R[x_batch[i, j]] += grad_emb[i, j]

    return loss, grad_R, grad_W_h, grad_b_h, grad_W_o, grad_b_o

def update_params(params, grads, velocities, lr, momentum):
    for p, g, v in zip(params, grads, velocities):
        v[...] = momentum * v - lr * g
        p += v

def compute_loss(valx, vald, R, W_h, b_h, W_o, b_o):
    B = valx.shape[0]
    emb = R[valx, :] 
    emb_flat = emb.reshape(B, 3*D)
    h_in = emb_flat @ W_h + b_h
    h = sigmoid(h_in)
    logits = h @ W_o + b_o
    loss, _ = cross_entropy_loss_and_grad(logits, vald)
    
    return loss

def predict(x, R, W_h, b_h, W_o, b_o):
    B = x.shape[0]
    emb = R[x, :]
    emb_flat = emb.reshape(B, 3*D)
    h_in = emb_flat @ W_h + b_h
    h = sigmoid(h_in)
    logits = h @ W_o + b_o
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    return preds

best_val_loss = float('inf')
epochs_no_improve = 0

num_train = len(trainx)
num_batches = num_train // batch_size

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(max_epochs):
    perm = np.random.permutation(num_train)
    trainx = trainx[perm]
    traind = traind[perm]
    
    total_train_loss = 0.0
    train_correct = 0  
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch = trainx[start:end]
        d_batch = traind[start:end]

        loss, grad_R, grad_W_h, grad_b_h, grad_W_o, grad_b_o = forward_backward(
            x_batch, d_batch, R, W_h, b_h, W_o, b_o
        )
        total_train_loss += loss * (end - start)

        update_params(
            [R, W_h, b_h, W_o, b_o],
            [grad_R, grad_W_h, grad_b_h, grad_W_o, grad_b_o],
            [v_R, v_W_h, v_b_h, v_W_o, v_b_o],
            learning_rate, momentum_rate
        )
        
        preds = predict(x_batch, R, W_h, b_h, W_o, b_o)
        train_correct += np.sum(preds == d_batch)
    
    avg_train_loss = total_train_loss / num_train
    train_accuracy = train_correct / num_train

    # valid losss
    val_loss = compute_loss(valx, vald, R, W_h, b_h, W_o, b_o)
    
    val_preds = predict(valx, R, W_h, b_h, W_o, b_o)
    val_accuracy = np.mean(val_preds == vald)

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{max_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Early stopping algortihm 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = (R.copy(), W_h.copy(), b_h.copy(), W_o.copy(), b_o.copy())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break




R, W_h, b_h, W_o, b_o = best_params


test_loss = compute_loss(testx, testd, R, W_h, b_h, W_o, b_o)


test_preds = predict(testx, R, W_h, b_h, W_o, b_o)
test_acc = np.mean(test_preds == testd)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

################## part b ###########################



with h5py.File("data2.h5", 'r') as file:
    testx = np.array(file['testx'])

    words_dataset = [word.decode('utf-8') if isinstance(word, bytes) else word for word in file['words']]
    index_to_word = {i: word for i, word in enumerate(words_dataset)}

def get_probs(x, R, W_h, b_h, W_o, b_o):
   
    B = x.shape[0]
    emb = R[x, :]                 
    emb_flat = emb.reshape(B, 3*D) 
    h_in = emb_flat @ W_h + b_h    
    h = sigmoid(h_in)
    logits = h @ W_o + b_o        
    probs = softmax(logits)       
    return probs


num_samples = 5

sample_indices = np.random.choice(len(testx), size=num_samples, replace=False)


sample_x = testx[sample_indices] 
probs_all = get_probs(sample_x, R, W_h, b_h, W_o, b_o)  # shape (5, 250)

for i, trigram_indices in enumerate(sample_x):
    p = probs_all[i]
    
    top10_indices = np.argsort(p)[::-1][:10]
    top10_probs   = p[top10_indices]
    
    trigram_words = [index_to_word[idx] for idx in trigram_indices]
    
    predicted_words = [index_to_word[idx] for idx in top10_indices]
    
    print(f"\nSample {i+1}")
    print("Context (Trigram):", trigram_words)
    print("Top 10 Predictions for Next Word:")
    for w, prob_val in zip(predicted_words, top10_probs):
        print(f"  {w:<15} p={prob_val:.4f}")



