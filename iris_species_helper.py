import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from itertools import cycle


def torch_loader(batch_size=1, shuffle=False):
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
    y = iris.target  # Labels (species)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Function to create batches
    def create_batches(data, labels, batch_size, shuffle=True):
        num_samples = data.shape[0]
        if shuffle:
            indices = np.random.permutation(num_samples)
        else:
            indices = np.arange(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            yield data[batch_indices], labels[batch_indices]
    train_batches = cycle(create_batches(X_train, y_train, batch_size=batch_size, shuffle=shuffle))
    total_batches = (len(X_train) + batch_size - 1) // batch_size

    return train_batches, (X_train, y_train), (X_test, y_test), total_batches


if __name__ == "__main__":
    batch_size = 2
    train_loader, train, test, total_batches = torch_loader(batch_size)
    print(total_batches)
    # Example: Iterate through the DataLoader
    for batch_x, batch_y in train_loader:
        
        print("Batch X:", batch_x, type(batch_x), batch_x[0].shape)
        print("Batch y:", batch_y, type(batch_y))
        for x in batch_x:
            print(x, type(x))

        break  # Just show the first batch