import matplotlib.pyplot as plt
from keras.callbacks import History

# %% plot accuracy
def plot_training(history: History):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(len(history.epoch))], history.history['acc'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Training accuracy with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training accuracy", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


# %% plot loss
def plot_loss(history: History):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(len(history.epoch))], history.history['loss'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Training loss with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


# %% plot validation accuracy
def plot_validation(history: History):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(len(history.epoch))], history.history['val_acc'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Validation accuracy with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Validation accuracy", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


# %% plot validation loss
def plot_validation_loss(history: History):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(len(history.epoch))], history.history['val_loss'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Validation loss with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Validation loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


# %% plot validation loss
def plot_loss_combine(history_gen: History, history_dis: History):
    plt.figure(figsize=(7, 4))
    plt.plot([i + 1 for i in range(len(history_gen.epoch))], history_gen.history['loss'], '-o', lw=2, markersize=9,
             color='blue')
    plt.plot([i + 1 for i in range(len(history_dis.epoch))], history_dis.history['loss'], '-o', lw=2, markersize=9,
             color='orange')
    plt.grid(True)
    plt.legend(['Generator', 'Discriminator'])
    plt.title("Validation loss with epochs\n", fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training loss", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
