import os
import numpy as np
from copy import deepcopy

# corrupt the dataset
def corrupt_dataset(dataset, noise_type, noise_rate, nb_classes=10, model_dir=''):
    
    noisy_dataset = deepcopy(dataset)

    if noise_type == 'none':
        return noisy_dataset

    elif noise_type == 'label_symmetric':
        P = np.ones([nb_classes, nb_classes]) * noise_rate / nb_classes + np.identity(nb_classes) * (1 - noise_rate)
        #P = (noise_rate / (nb_classes - 1)) * P
        #for i in range(nb_classes):
        #    P[i, i] = 1 - noise_rate
        print('label transition matrix: \n', P)
        print('noise rate: ', noise_rate)
        # generate noisy labels according to P
        flipper = np.random.RandomState()
        if not hasattr(noisy_dataset, 'targets'):
            labels = np.asarray(noisy_dataset.train_labels)
        else:
            labels = np.asarray(noisy_dataset.targets)
        true_labels = labels.copy()

        for i in range(labels.shape[0]):
            l = labels[i]
            nl = flipper.multinomial(1, P[l, :])
            labels[i] = np.where(nl == 1)[0]

        actual_P = np.zeros([nb_classes, nb_classes])
        for i in range(nb_classes):
            for j in range(nb_classes):
                actual_P[i, j] = np.sum((true_labels == i) & (labels == j)) / np.sum(true_labels == i)
        print('actual label transition matrix: \n', actual_P)
        print('actual noise rate: ', np.mean(true_labels != labels))

        np.save(os.path.join(model_dir, 'noisy_label.npy'), labels)
        
        if not hasattr(noisy_dataset, 'targets'):
            noisy_dataset.train_labels = labels.tolist()
        else:
            noisy_dataset.targets = labels.tolist()
        return noisy_dataset

    elif noise_type == 'shuffle_pixel':
        train_images = noisy_dataset.data if hasattr(noisy_dataset, 'data') else noisy_dataset.train_data
        n_train = len(train_images)
        n_rand = int(noise_rate * n_train)
        randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
        print('Randomizing {} out of {} by {}.'.format(n_rand, n_train, noise_type))
        randomized_images = train_images[randomize_indices].copy()
        n, h, w, c = randomized_images.shape
        randomized_images = randomized_images.transpose(1, 2, 0, 3).reshape(h*w, n*c)
        shuffle_matrix = np.arange(h*w, dtype=np.int)
        np.random.shuffle(shuffle_matrix)
        randomized_images = randomized_images[shuffle_matrix]
        randomized_images = randomized_images.reshape(h, w, n, c).transpose(2, 0, 1, 3)
        train_images[randomize_indices] = randomized_images
        if hasattr(noisy_dataset, 'data'):
            noisy_dataset.data = train_images
        else:
            noisy_dataset.train_data = train_images
        return noisy_dataset

    elif noise_type == 'random_pixel':
        train_images = noisy_dataset.data if hasattr(noisy_dataset, 'data') else noisy_dataset.train_data
        n_train = len(train_images)
        n_rand = int(noise_rate * n_train)
        randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
        print('Randomizing {} out of {} by {}.'.format(n_rand, n_train, noise_type))
        randomized_images = train_images[randomize_indices].copy()
        n, h, w, c = randomized_images.shape
        for i in range(n):
            shuffle_matrix = np.arange(h*w, dtype=np.int)
            np.random.shuffle(shuffle_matrix)
            randomized_images[i] = randomized_images[i].reshape(h*w, c)[shuffle_matrix].reshape(h, w, c)
        train_images[randomize_indices] = randomized_images
        if hasattr(noisy_dataset, 'data'):
            noisy_dataset.data = train_images
        else:
            noisy_dataset.train_data = train_images
        return noisy_dataset

    elif noise_type == 'gaussian':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_images = noisy_dataset.data if hasattr(noisy_dataset, 'data') else noisy_dataset.train_data
        n_train = len(train_images)
        n_rand = int(noise_rate * n_train)
        randomize_indices = np.random.choice(range(n_train), size=n_rand, replace=False)
        print('Randomizing {} out of {} by {}.'.format(n_rand, n_train, noise_type))
        randomized_images = np.random.randn(*train_images[randomize_indices].shape)
        randomized_images *= np.asarray(std)
        randomized_images += np.asarray(mean)
    
        randomized_images = np.clip(randomized_images, a_min=0, a_max=1)
        train_images[randomize_indices] = randomized_images
        if hasattr(noisy_dataset, 'data'):
            noisy_dataset.data = train_images
        else:
            noisy_dataset.train_data = train_images
        return noisy_dataset


    else:
        raise NotImplementedError
