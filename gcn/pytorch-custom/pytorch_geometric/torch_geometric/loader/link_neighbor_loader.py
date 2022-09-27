from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.neighbor_loader import NeighborSampler
from torch_geometric.loader.utils import filter_data, filter_hetero_data
from torch_geometric.typing import InputEdges, NumNeighbors, OptTensor


class LinkNeighborSampler(NeighborSampler):
    def __init__(self, data, *args, neg_sampling_ratio: float = 0.0, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.neg_sampling_ratio = neg_sampling_ratio

        if issubclass(self.data_cls, Data):
            self.num_src_nodes = self.num_dst_nodes = data.num_nodes
        else:
            self.num_src_nodes = data[self.input_type[0]].num_nodes
            self.num_dst_nodes = data[self.input_type[-1]].num_nodes

    def _create_label(self, edge_label_index, edge_label):
        device = edge_label_index.device

        num_pos_edges = edge_label_index.size(1)
        num_neg_edges = int(num_pos_edges * self.neg_sampling_ratio)

        if num_neg_edges == 0:
            return edge_label_index, edge_label

        if edge_label is None:
            edge_label = torch.ones(num_pos_edges, device=device)
        else:
            assert edge_label.dtype == torch.long
            edge_label = edge_label + 1

        neg_row = torch.randint(self.num_src_nodes, (num_neg_edges, ))
        neg_col = torch.randint(self.num_dst_nodes, (num_neg_edges, ))
        neg_edge_label_index = torch.stack([neg_row, neg_col], dim=0)

        neg_edge_label = edge_label.new_zeros((num_neg_edges, ) +
                                              edge_label.size()[1:])

        edge_label_index = torch.cat([
            edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        edge_label = torch.cat([edge_label, neg_edge_label], dim=0)

        return edge_label_index, edge_label

    def __call__(self, query: List[Tuple[Tensor]]):
        query = [torch.tensor(s) for s in zip(*query)]
        if len(query) == 2:
            edge_label_index = torch.stack(query, dim=0)
            edge_label = None
        else:
            edge_label_index = torch.stack(query[:2], dim=0)
            edge_label = query[2]

        edge_label_index, edge_label = self._create_label(
            edge_label_index, edge_label)

        if issubclass(self.data_cls, Data):
            sample_fn = torch.ops.torch_sparse.neighbor_sample

            query_nodes = edge_label_index.view(-1)
            query_nodes, reverse = query_nodes.unique(return_inverse=True)
            edge_label_index = reverse.view(2, -1)

            node, row, col, edge = sample_fn(
                self.colptr,
                self.row,
                query_nodes,
                self.num_neighbors,
                self.replace,
                self.directed,
            )

            return node, row, col, edge, edge_label_index, edge_label

        elif issubclass(self.data_cls, HeteroData):
            sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample

            if self.input_type[0] != self.input_type[-1]:
                query_src = edge_label_index[0]
                query_src, reverse_src = query_src.unique(return_inverse=True)
                query_dst = edge_label_index[1]
                query_dst, reverse_dst = query_dst.unique(return_inverse=True)
                edge_label_index = torch.stack([reverse_src, reverse_dst], 0)
                query_node_dict = {
                    self.input_type[0]: query_src,
                    self.input_type[-1]: query_dst,
                }
            else:  # Merge both source and destination node indices:
                query_nodes = edge_label_index.view(-1)
                query_nodes, reverse = query_nodes.unique(return_inverse=True)
                edge_label_index = reverse.view(2, -1)
                query_node_dict = {self.input_type[0]: query_nodes}

            node_dict, row_dict, col_dict, edge_dict = sample_fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                query_node_dict,
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )

            return (node_dict, row_dict, col_dict, edge_dict, edge_label_index,
                    edge_label)


class LinkNeighborLoader(torch.utils.data.DataLoader):
    r"""A link-based data loader derived as an extension of the node-based
    :class:`torch_geometric.loader.NeighborLoader`.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, this loader first selects a sample of edges from the
    set of input edges :obj:`edge_label_index` (which may or not be edges in
    the original graph) and then constructs a subgraph from all the nodes
    present in this list by sampling :obj:`num_neighbors` neighbors in each
    iteration.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = LinkNeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            edge_label_index=data.edge_index,
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
                 train_mask=[1368], val_mask=[1368], test_mask=[1368],
                 edge_label_index=[2, 128])

    It is additionally possible to provide edge labels for sampled edges, which
    are then added to the batch:

    .. code-block:: python

        loader = LinkNeighborLoader(
            data,
            num_neighbors=[30] * 2,
            batch_size=128,
            edge_label_index=data.edge_index,
            edge_label=torch.ones(data.edge_index.size(1))
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
                 train_mask=[1368], val_mask=[1368], test_mask=[1368],
                 edge_label_index=[2, 128], edge_label=[128])

    The rest of the functionality mirrors that of
    :class:`~torch_geometric.loader.NeighborLoader`, including support for
    heterogenous graphs.

    .. note::
        :obj:`neg_sampling_ratio` is currently implemented in an approximate
        way, *i.e.* negative edges may contain false negatives.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The edge indices for which neighbors are sampled to create
            mini-batches.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and corresponding edge indices.
            (default: :obj:`None`)
        edge_label (Tensor): The labels of edge indices for which neighbors are
            sampled. Must be the same length as the :obj:`edge_label_index`.
            If set to :obj:`None` then no labels are returned in the batch.
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            edges to the number of positive edges.
            If :obj:`edge_label` does not exist, it will be automatically
            created and represents a binary classification task
            (:obj:`1` = edge, :obj:`0` = no edge).
            If :obj:`edge_label` exists, it has to be a categorical label from
            :obj:`0` to :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges.
            Note that returned labels are of type :obj:`torch.float` for binary
            classification (to facilitate the ease-of-use of
            :meth:`F.binary_cross_entropy`) and of type
            :obj:`torch.long` for multi-class classification (to facilitate the
            ease-of-use of :meth:`F.cross_entropy`). (default: :obj:`0.0`).
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: NumNeighbors,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        transform: Callable = None,
        neighbor_sampler: Optional[LinkNeighborSampler] = None,
        neg_sampling_ratio: float = 0.0,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = neighbor_sampler
        self.neg_sampling_ratio = neg_sampling_ratio

        edge_type, edge_label_index = get_edge_label_index(
            data, edge_label_index)

        if neighbor_sampler is None:
            self.neighbor_sampler = LinkNeighborSampler(
                data, num_neighbors, replace, directed, edge_type,
                share_memory=kwargs.get('num_workers', 0) > 0,
                neg_sampling_ratio=self.neg_sampling_ratio)

        super().__init__(Dataset(edge_label_index, edge_label),
                         collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, out: Any) -> Union[Data, HeteroData]:
        if isinstance(self.data, Data):
            node, row, col, edge, edge_label_index, edge_label = out
            data = filter_data(self.data, node, row, col, edge,
                               self.neighbor_sampler.perm)
            data.edge_label_index = edge_label_index
            if edge_label is not None:
                data.edge_label = edge_label

        elif isinstance(self.data, HeteroData):
            (node_dict, row_dict, col_dict, edge_dict, edge_label_index,
             edge_label) = out
            data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                      edge_dict,
                                      self.neighbor_sampler.perm_dict)
            edge_type = self.neighbor_sampler.input_type
            data[edge_type].edge_label_index = edge_label_index
            if edge_label is not None:
                data[edge_type].edge_label = edge_label

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


class Dataset(torch.utils.data.Dataset):
    def __init__(self, edge_label_index: Tensor, edge_label: OptTensor = None):
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label

    def __getitem__(self, idx: int) -> Tuple[int]:
        if self.edge_label is None:
            return self.edge_label_index[0, idx], self.edge_label_index[1, idx]
        else:
            return (self.edge_label_index[0, idx],
                    self.edge_label_index[1, idx], self.edge_label[idx])

    def __len__(self) -> int:
        return self.edge_label_index.size(1)


def get_edge_label_index(
    data: Union[Data, HeteroData],
    edge_label_index: InputEdges,
) -> Tuple[Optional[str], Tensor]:
    edge_type = None
    if isinstance(data, Data):
        if edge_label_index is None:
            return None, data.edge_index
        return None, edge_label_index

    assert edge_label_index is not None
    assert isinstance(edge_label_index, (list, tuple))

    if isinstance(edge_label_index[0], str):
        edge_type = edge_label_index
        edge_type = data._to_canonical(*edge_type)
        assert edge_type in data.edge_types
        return edge_type, data[edge_type].edge_index

    assert len(edge_label_index) == 2

    edge_type, edge_label_index = edge_label_index
    edge_type = data._to_canonical(*edge_type)
    assert edge_type in data.edge_types

    if edge_label_index is None:
        return edge_type, data[edge_type].edge_index

    return edge_type, edge_label_index
