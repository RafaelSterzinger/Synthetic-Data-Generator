import matplotlib.pyplot as plt


# %% plot accuracy
def plot_training(episodes: int, history):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(episodes)], history.history['acc'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Training accuracy with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training accuracy", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# %% plot loss
def plot_loss(episodes: int, history):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(episodes)], history.history['loss'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Training loss with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# %% plot validation accuracy
def plot_validation(episodes: int, history):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(episodes)], history.history['val_acc'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Validation accuracy with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training accuracy", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

# %% plot validation loss
def plot_validation_loss(episodes: int, history):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(episodes)], history.history['val_loss'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Validation loss with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Validation loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
