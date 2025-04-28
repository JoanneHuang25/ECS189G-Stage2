from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

#---- Multi-Layer Perceptron script for MNIST digit classification ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('mnist', 'MNIST dataset')
    data_obj.dataset_source_folder_path = 'data/stage_2_data/'
    
    method_obj = Method_MLP('multi-layer perceptron', 'MLP model for MNIST')
    method_obj.max_epoch = 500
    
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = 'result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'mnist_prediction_result'
    
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    print('Loading training data...')
    train_data = data_obj.load(train=True)
    print('Loading testing data...')
    test_data = data_obj.load(train=False)
    
    # Pass the loaded data to the method
    method_obj.data = {'train': train_data, 'test': test_data}
    
    # Run the method to train and test the model
    print('Training and testing the model...')
    result = method_obj.run()
    
    # Save results
    result_obj.data = result
    result_obj.save()
    
    print('************ Overall Performance ************')
    metrics = result['metrics']
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')
    
    # Create and save the loss plot
    print('Creating training loss plot...')
    loss_values = result['loss_values']
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_values)), loss_values, marker='', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('../../result/stage_2_result/training_loss_plot.png')
    plt.show()
    
    print('************ Finish ************')
    # ------------------------------------------------------
    

    