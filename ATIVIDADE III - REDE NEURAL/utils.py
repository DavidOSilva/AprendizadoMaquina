from colorama import Fore, Style
import matplotlib.pyplot as plt
from SimplesMLP import SimplesMLP
from PIL import Image
import numpy as np
import torch

def printColorido(message, color='green', end='\n'):
    color_map = { # Dicionário para mapear as cores fornecidas ao colorama
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE
    }
    if color in color_map: color_code = color_map[color] # Verifica se a cor fornecida está no dicionário.
    else:
        print("Cor inválida. Usando a cor padrão.")
        color_code = Fore.RESET
    print(f"{color_code}{message}{Style.RESET_ALL}", end=end) # Imprime a mensagem com a cor desejada.

def contarClasses(y):
    classes = {}
    for index, linha in y.iterrows():
        linha = linha.to_dict() # Teremos algo assim tem cada linha: {'num': 0}
        classe = list(linha.values())[0] # Desestrutura diretamente o valor único da linha.
        if classe not in classes: classes[classe] = 1
        else: classes[classe] += 1
    return classes

def obterIncompletude(df):
    linhasIncompletas = []
    incompletos = df.isnull().sum()
    for atributo, faltantes in incompletos.items(): # Obtém os índices das linhas com valores ausentes.
        if faltantes > 0: linhasIncompletas.extend(df.index[df[atributo].isnull()].tolist())
    printColorido(f"QUANTIDADE: ", "yellow", "")
    print(len(linhasIncompletas))
    printColorido(f"INDICES: ", "yellow", "")
    print(linhasIncompletas)
    return linhasIncompletas
    
def obterRedundancias(x):
    duplicadas = x.index[x.duplicated(keep="first")].tolist()
    printColorido(f"QUANTIDADE: ", "yellow", "")
    print(len(duplicadas))
    printColorido(f"INDICES: ", "yellow", "")
    print(duplicadas)
    return duplicadas

class EarlyStopper:
    def __init__(self, patience=14, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience: return True
        return False
    
def exibir(modo, loss_array, acc_array, f1_array, precision_array, recall_array, timer_start, timer_end):
    tipos = ['TREINAMENTO:', 'VALIDAÇÃO:  ']
    escolhido = 'SEM INFORMAÇÃO'
    if(modo == 'treinamento'): escolhido = tipos[0]
    elif(modo == 'teste' or modo == 'validação'): escolhido = tipos[1]
    else: print('SELECIONE UM MODO INVÁLIDO')
        
    print(
        f'{escolhido}    LOSS: {"{:.4f}".format(loss_array.mean())}  ',
        f'ACCURACY: {"{:.2f}".format(acc_array.mean()*100)}%  ',
        f'F-1: {"{:.2f}".format(f1_array.mean()*100)}%  ',
        f'PRECISÃO: {"{:.2f}".format(precision_array.mean()*100)}%  ',
        f'RECALL: {"{:.2f}".format(recall_array.mean()*100)}%  ',
        f'TEMPO: {"{:.2f}".format(timer_end-timer_start)}s'
         )

def plotarMetricas(foldMetric):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4), dpi=300) # Plotando o gráfico 1x5
    titles = ['Perda', 'Acurácia', 'F1-Score', 'Precisão', 'Revocação']
    for m in range(5):
        axs[m].plot(foldMetric[0][m], label='Treinamento', color='darkorchid', linewidth = 2)
        axs[m].plot(foldMetric[1][m], label='Validação', color='plum', linewidth = 2)
        axs[m].set_title(titles[m])
        axs[m].set_xlabel('Épocas')
        axs[m].legend()
        
    for ax in axs:
        ax.grid(True, linestyle = '--', linewidth = 1, alpha=0.5)

    # Ajustando o layout e mostrando o gráfico
    plt.tight_layout()
    plt.show()
    
def plotarPredicoes(xTeste, predicoes, rotulos, numPlots=10, inicio=0):
    # Calculando o número de linhas e colunas para os subplots
    cols = min(10, numPlots)  # Máximo de colunas para evitar subplots muito largos
    rows = (numPlots + cols - 1) // cols  # Calcula o número de linhas necessárias
    
    # Ajustando o tamanho da figura com base no número de subplots
    figsize = (cols * 2.5, rows * 2.5)  # Cada subplot terá 3x3 polegadas
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()  # Flatten para facilitar o acesso aos subplots

    for i in range(inicio, numPlots + inicio):
        pred, lbl = np.argmax(predicoes[i].cpu()), rotulos[i]  # Obtendo a predição e o rótulo
        status = "✔" if pred == lbl else "✘"  # Determinando se a predição está correta
        
        axes[i - inicio].set_title(f"Pred.: {pred}   Rótulo: {lbl}   {status}", pad=20)  # Título com a predição, o rótulo e o status
        axes[i - inicio].imshow(xTeste[i].cpu().reshape(8, 8), cmap="binary")  # Plotando a imagem
        axes[i - inicio].grid(True, linestyle = '--', linewidth = 1, alpha=0.3)
        
    # Desligando os subplots extras caso haja menos plots que o espaço alocado
    for j in range(numPlots, len(axes)): axes[j].axis("off")

    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()  # Ajustando automaticamente o layout
    plt.show()
    
def inferirManuscrito(imagepath, modelpath, device):
    # Carregando a imagem
    img = Image.open(imagepath).convert('L')
    imgResized = img.resize((8, 8))
    
    # Convertendo a imagem para um array
    imgArray = np.array(imgResized).flatten()
    imgArray = np.round(imgArray / 16.0)  # Garantindo valores entre 0 e 16
    
    # Convertendo a imagem para tensor
    imgTensor = torch.tensor(imgArray, dtype=torch.float32).unsqueeze(0)  # Adicionando dimensão extra para batch
    
    # Instancia e carrega pesos ao modelo, assim não é necessário treinar novamente.
    model = SimplesMLP(inputChannels=64, numClasses=10).to(device)
    model.load_state_dict(torch.load(modelpath, weights_only=True, map_location=torch.device('cpu')))
    
    # Realizando a predição
    model.eval()
    with torch.no_grad(): 
        ypred = model(imgTensor.to(device))
    
    # Obtendo os 10 valores de probabilidade para cada dígito
    probabilidades = torch.nn.functional.softmax(ypred, dim=1).cpu().numpy().flatten()
    
    return imgResized, probabilidades, probabilidades.argmax()

def plotManuscritoComDígitos(imagepath, modelpath, device):
    imgResized, probs, digito = inferirManuscrito(imagepath, modelpath, device)

    # Criando a figura com 2 subgráficos (um para a imagem original e outro para os 10 dígitos)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Exibindo a imagem original
    axs[0].imshow(imgResized, cmap='binary')
    axs[0].set_title('Manuscrito 8x8', pad=30)
    axs[0].axis('off')  # Remover eixos
    axs[0].set_xlim(-1, 9)  # Ajustando os limites de x
    axs[0].set_ylim(9, -1)  # Ajustando os limites de y

    # Criando um gráfico para os 10 dígitos com intensidade baseada nas probabilidades
    digit_colors = plt.cm.RdPu(probs / np.max(probs))  # Usando colormap 'RdPu' para tons de vermelho e roxo
    axs[1].barh(np.arange(10), probs, color='plum')
    axs[1].set_title('Probabilidade dos Dígitos', pad=30)
    axs[1].set_yticks(np.arange(10))
    axs[1].grid(True, axis='x', zorder=0, linestyle = '--', linewidth = 1, alpha=0.5) 
    
    plt.tight_layout()
    plt.show()


