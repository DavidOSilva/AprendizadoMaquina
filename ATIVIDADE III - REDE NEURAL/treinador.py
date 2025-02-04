from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from utils import *
import time
from pathlib import Path

def calcularMetricas(ypred, rotulo, num_classes=10):
    preds, target = ypred.detach().cpu(), rotulo.detach().cpu()

    metric_acc = MulticlassAccuracy(num_classes=num_classes, average='macro')
    metric_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
    metric_prec = MulticlassPrecision(num_classes=num_classes, average='macro')
    metric_recall = MulticlassRecall(num_classes=num_classes, average='macro')

    acc = metric_acc(preds, target)
    f1 = metric_f1(preds, target)
    prec = metric_prec(preds, target)
    recall = metric_recall(preds, target)
    
    return acc.item(), f1.item(), prec.item(), recall.item()


def treinar(modelo, loader, criterion, optimizer, device):
    modelo.train() #É importante alterar a rede neural para o modo adequado, seja para treinamento ou validação.
    start = time.time() # Para incrementar, vamos exibir o tempo que durou cada época de treinamento, assim vamos iniciar a contagem.
    lossEpocas, accEpocas, f1Epocas,  precEpocas, recallEpocas = [], [], [], [], []

    for batch_idx, (dado, rotulo) in enumerate(loader):
        dado, rotulo = dado.to(device), rotulo.to(device)
        ypred = modelo(dado)

        loss = criterion(ypred, rotulo) # Calcula a perda
        acc, f1, prec, recall = calcularMetricas(ypred, rotulo) # Obtendo as metricas.
        lossEpocas.append(loss.detach().item()), accEpocas.append(acc), f1Epocas.append(f1), precEpocas.append(prec), recallEpocas.append(recall)

        #Vejamos agora o Backpropagation:
        optimizer.zero_grad() # Precisamos zerar o gradiente para garantir um funcionamento adequado do nosso treinamento a cada nova iteração.
        loss.backward() # Derivando e calculando o gradiente.
        optimizer.step() # Atualizando os pesos usando os resultados como indicativos.
    
    lossEpocas, accEpocas, f1Epocas, precEpocas, recallEpocas = np.array(lossEpocas), np.array(accEpocas), np.array(f1Epocas), np.array(precEpocas), np.array(recallEpocas)
    end = time.time() # Finalizando a contagem.
    exibir('treinamento', lossEpocas, accEpocas, f1Epocas, precEpocas, recallEpocas, start, end) #Exibindo as métricas.
    return lossEpocas.mean(), accEpocas.mean(), f1Epocas.mean(), precEpocas.mean(), recallEpocas.mean() #Vamos retornar a média das perdas para acompanhar a convergência.


def validar(modelo, loader, criterion, device):
    modelo.eval()
    start = time.time()
    lossEpocas, accEpocas, f1Epocas,  precEpocas, recallEpocas = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (dado, rotulo) in enumerate(loader):
            dado, rotulo = dado.to(device), rotulo.to(device)
            ypred = modelo(dado)

            loss = criterion(ypred, rotulo)
            acc, f1, prec, recall = calcularMetricas(ypred, rotulo) #Obtendo as metricas.
            lossEpocas.append(loss.detach().item()), accEpocas.append(acc), f1Epocas.append(f1), precEpocas.append(prec), recallEpocas.append(recall)
    
    lossEpocas, accEpocas, f1Epocas, precEpocas, recallEpocas = np.array(lossEpocas), np.array(accEpocas), np.array(f1Epocas), np.array(precEpocas), np.array(recallEpocas)
    end = time.time()
    exibir('treinamento', lossEpocas, accEpocas, f1Epocas, precEpocas, recallEpocas, start, end) #Exibindo as métricas.
    return lossEpocas.mean(), accEpocas.mean(), f1Epocas.mean(), precEpocas.mean(), recallEpocas.mean()


def treinador(fold, model, criterion, optimizer, scheduler, loaders, device, numMaxEpocas=200):
    tLoss, tAcc, tF1, tPrec, tRec,   vLoss, vAcc, vF1, vPrec, vRec = [], [], [], [], [],    [], [], [], [], []
    earlyStopper, melhorScore = EarlyStopper(), 0
    for epoca in range(numMaxEpocas):
        
        print(f'--------------------------------------------------- ÉPOCA {epoca+1} ----------------------------------------------------')
        loss, acc, f1, prec, rec = treinar(model, loaders[0], criterion, optimizer, device) # Treinar o modelo
        tLoss.append(loss)
        tAcc.append(acc)
        tF1.append(f1)
        tPrec.append(prec)
        tRec.append(rec)
        
        loss, acc, f1, prec, rec = validar(model, loaders[1], criterion, device) # Avaliar no conjunto de validação
        vLoss.append(loss)
        vAcc.append(acc)
        vF1.append(f1)
        vPrec.append(prec)
        vRec.append(rec)
        
        score = (vAcc[epoca] + vF1[epoca]*1.3 + vPrec[epoca] + vRec[epoca])/vLoss[epoca]
        if(score > melhorScore ): melhorScore, melhorPesos, melhorEpoca  = score, model.state_dict(), epoca + 1
        
        scheduler.step(loss) # Verifica se a taxa de aprendizado deve ser reduzida.
        if earlyStopper.early_stop(loss): break #Verificando se o treinamento deve ser encerrado.
    printColorido(f'\nO MELHOR DESEMPENHO FOI ALCANÇADO NA ÉPOCA {melhorEpoca}', 'blue')
    if epoca < numMaxEpocas - 1: printColorido(f'\nO TREINAMENTO PRECISOU SER INTERROMPIDO NA ÉPOCA {epoca+1}!', 'yellow')
    else: printColorido('\nO TREINAMENTO FOI FINALIZADO COM SUCESSO!')
    ptsTreino, ptsVal = [tLoss, tAcc, tF1, tPrec, tRec], [vLoss, vAcc, vF1, vPrec, vRec]
    
    model.load_state_dict(melhorPesos) # Carregar os melhores pesos.
    pastaFold = Path(f"weights/fold_{fold}")
    pastaFold.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pastaFold/'weights.pth')
    
    return model, [ptsTreino, ptsVal, melhorEpoca]