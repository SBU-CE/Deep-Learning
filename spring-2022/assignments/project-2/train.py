from model import *
from torch_helper import *

import matplotlib.pyplot as plt
import numpy as np


def train(args, x_train, y_train, x_test, y_test, colours, model_mode=None, model=None):
    # Set the maximum number of threads to prevent crash in Teaching Labs
    #####################################################################################
    # TODO: Implement this function to train model and consider the below items         #
    # 0. read the utils file and use 'process' and 'get_rgb_cat' to get x and y for     #
    #    test and train dataset                                                         #
    # 1. Create train and test data loaders with respect to some hyper-parameters       #
    # 2. Get an instance of your 'model_mode' based on 'model_mode==base' or            #
    #    'model_mode==U-Net'.                                                           #
    # 3. Define an appropriate loss function (cross entropy loss)                       #
    # 4. Define an optimizers with proper hyper-parameters such as (learning_rate, ...).#
    # 5. Implement the main loop function with n_epochs iterations which the learning   #
    #    and evaluation process occurred there.                                         #
    # 6. Save the model weights                                                         #
    # Hint: Modify the predicted output form the model, to use loss function in step 3  #
    #####################################################################################
    """
    Get colour categories given RGB values. This function doesn't
    actually do the work, instead it splits the work into smaller
    chunks that can fit into memory, and calls helper function
    _get_rgb_cat

    Args:
     model_mode: String
    Returns:
      model: trained model
    """
    torch.set_num_threads(5)

    # Numpy random seed
    np.random.seed(args.seed)

    # Save directory
    save_dir = "outputs/" + args.experiment_name

    print("Transforming data...")
    # Get X(grayscale images) and Y(the nearest Color to each pixel based on given color dictionary)
    train_rgb, train_grey = process(x_train, y_train, downsize_input=args.downsize_input, category_id=args.category_id)
    train_rgb_cat = rgb2label(train_rgb, colours, args.batch_size)
    test_rgb, test_grey = process(x_test, y_test, downsize_input=args.downsize_input, category_id=args.category_id)
    test_rgb_cat = rgb2label(test_rgb, colours, args.batch_size)

    # LOAD THE MODEL
    ##############################################################################################
    #                                            YOUR CODE                                       #
    ##############################################################################################

    # LOSS FUNCTION and Optimizer
    ##############################################################################################
    #                                            YOUR CODE                                       #
    ##############################################################################################

    # Create the outputs' folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_losses = []
    valid_losses = []
    valid_accs = []

    # Training loop
    for epoch in range(args.epochs):
        # Train the Model
        model.train()  # Change model to 'train' mode
        losses = []
        for i, (xs, ys) in enumerate(get_batch(train_grey,
                                               train_rgb_cat,
                                               args.batch_size)):
            # Convert numpy array to pytorch tensors
            images, labels = get_torch_vars(xs, ys, args.gpu)

            # Forward + Backward + Optimize
            ##############################################################################################
            #                                            YOUR CODE                                       #
            ##############################################################################################

        # Calculate and Print training loss for each epoch
        ##############################################################################################
        #                                            YOUR CODE                                       #
        ##############################################################################################

        # Evaluate the model
        ##############################################################################################
        #                                            YOUR CODE                                       #
        ##############################################################################################

        # Calculate and Print (validation loss, validation accuracy) for each epoch
        ##############################################################################################
        #                                            YOUR CODE                                       #
        ##############################################################################################

    # Plot training-validation curve
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir + "/training_curve.png")

    if args.checkpoint:
        print('Saving model...')
        torch.save(model.state_dict(), args.checkpoint)

    return model
