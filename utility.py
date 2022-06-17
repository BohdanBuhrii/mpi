import numpy as np

def get_random_matrix(N, M):
  # discrete uniform distribution on [-10, 10]
  return np.random.randint(19, size=(N, M)) + np.random.rand(N,M) - 10

def generate_data(N, M):
  # save random matrix and vector to the files
  np.save('matrix_multiplication/data/{}_{}.npy'.format(N, M), get_random_matrix(N, M))
  np.save('matrix_multiplication/data/{}.npy'.format(M), get_random_matrix(M, 1))

def verify_result(N, M):
  # load data
  matrix = np.load('matrix_multiplication/data/{}_{}.npy'.format(N, M), allow_pickle=True)
  vector = np.load('matrix_multiplication/data/{}.npy'.format(M), allow_pickle=True)
  result = np.load('matrix_multiplication/data/{}_{}_result.npy'.format(N, M), allow_pickle=True)

  # compute product with numpy
  correct_result = np.dot(matrix, vector)

  # compare results
  return np.allclose(result, correct_result)