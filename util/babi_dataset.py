from .common_function import *

class OneBatch():
    def __init__(self, adjacency_matrix_batch, annotation_batch, target_batch, init_input_batch):
        self.adjacency_matrix_batch = adjacency_matrix_batch
        self.annotation_batch = annotation_batch
        self.target_batch = target_batch
        self.init_input_batch = init_input_batch

class bAbIDataset():
    """
    Load bAbI tasks for GGNN
    """
    def __init__(self, data_path, task_id, is_train, batch_size=10, state_dim = 4, annotation_dim = 1):
        all_data = load_graphs_from_file(data_path)
        self.n_edge_types =  find_max_edge_id(all_data)
        self.n_tasks = find_max_task_id(all_data)
        self.node_num = find_max_node_id(all_data)

        all_task_train_data, all_task_val_data = split_set(all_data)

        if is_train:
            all_task_train_data = data_convert(all_task_train_data, annotation_dim)
            self.data = all_task_train_data[task_id]
        else:
            all_task_val_data = data_convert(all_task_val_data, annotation_dim)
            self.data = all_task_val_data[task_id]

        batch_spans = make_batches(len(self.data), batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_adjacency_matrix_batch = []
            cur_target_batch = []
            cur_annotation_batch = []
            cur_init_input_batch = []
            for i in range(batch_start, batch_end):
                cur_adjacency_matrix_batch.append(create_adjacency_matrix(self.data[i][0], self.node_num, self.n_edge_types))
                cur_target_batch.append(self.data[i][2] - 1)
                cur_annotation = self.data[i][1]
                cur_annotation_batch.append(cur_annotation)
                padding = np.zeros(shape=(self.node_num, state_dim - annotation_dim))
                cur_init_input_batch.append(np.concatenate([cur_annotation,padding],axis=-1))
            self.batches.append(OneBatch(cur_adjacency_matrix_batch,cur_annotation_batch,cur_target_batch,cur_init_input_batch))
        self.num_batch = len(self.batches)
        self.cur_batch_pointer = 0

    def next_batch(self):
        if self.cur_batch_pointer >= self.num_batch:
            self.cur_batch_pointer = 0
        cur_batch = self.batches[self.cur_batch_pointer]
        self.cur_batch_pointer += 1
        return cur_batch


