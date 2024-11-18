defmodule FilterEx.Utils do

  def tt(x) do
    Nx.transpose(x)
  end

  def zeros({m,n}) do
    Nx.broadcast(0.0, {m,n})
  end

  @doc ~s"""
    ensure z is a (dim_z, 1) shaped vector

    ## Examples

      iex> FilterEx.Utils.reshape_z(3.0, 1, 1)
      Nx.tensor([ 3.0 ], names: [:x])

      iex> FilterEx.Utils.reshape_z(3.0, 1, 2)
      Nx.tensor([[ 3.0 ]], names: [:x, :y])

      iex> FilterEx.Utils.reshape_z([3.0], 1, 1)
      Nx.tensor([ 3.0 ], names: [:x])

      iex> FilterEx.Utils.reshape_z([3.0], 1, 2)
      Nx.tensor([[3.0]], names: [:x, :y])

      iex> FilterEx.Utils.reshape_z([3.0,2.0], 2, 1)
      Nx.tensor([3.0, 2.0], names: [:x])

      iex> FilterEx.Utils.reshape_z([3.0,2.0], 2, 2)
      Nx.tensor([[3.0], [2.0]], names: [:x, :y])

  """
  def reshape_z(z, dim_z, ndim) do
    # IO.inspect(z, label: "reshape z::")
    # IO.inspect({dim_z, ndim}, label: "reshape {dim_z, ndim}::")

    z = if is_number(z) do Nx.broadcast(z, {1,1}) else z |> Nx.tensor() end

    z = case z |> Nx.shape() do
      {n} -> z |> Nx.reshape({1,n})
      _other -> z
    end

    z = z |> Nx.tensor(names: [:y, :x])
    # IO.inspect(z, label: Zz)

    z = if z |> Nx.shape() |> elem(1) == dim_z do z |> tt() else z end
    # IO.inspect(z, label: Zzz)

    if Nx.shape(z) != {dim_z, 1} do
        raise %ArgumentError{ message: "z (shape #{Nx.shape(z)}) must be convertible to shape (#{dim_z}, 1)" }
    end

    z = if ndim == 1 do z[y: 0] else z end
    # IO.inspect(z, label: Zzza)

    z = if ndim == 0 do z[0][0] else z end
    # IO.inspect(z, label: Zzzb)

    z
  end

  @deprecated
  @doc """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.

    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    Parameters
    -----------

    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)

    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations

    var : float, default=1.0
        variance in the noise

    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.

    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)

        [x x' x'' y y' y'']

        whereas `False` interleaves the dimensions

        [x y z x' y' z' x'' y'' z'']


    Examples
    --------
    >>> # constant velocity model in a 3D world with a 10 Hz update rate
    >>> Q_discrete_white_noise(2, dt=0.1, var=1., block_size=3)
    array([[0.000025, 0.0005  , 0.      , 0.      , 0.      , 0.      ],
           [0.0005  , 0.01    , 0.      , 0.      , 0.      , 0.      ],
           [0.      , 0.      , 0.000025, 0.0005  , 0.      , 0.      ],
           [0.      , 0.      , 0.0005  , 0.01    , 0.      , 0.      ],
           [0.      , 0.      , 0.      , 0.      , 0.000025, 0.0005  ],
           [0.      , 0.      , 0.      , 0.      , 0.0005  , 0.01    ]])

    References
    ----------

    Bar-Shalom. "Estimation with Applications To Tracking and Navigation".
    John Wiley & Sons, 2001. Page 274.
    """
  def q_discrete_white_noise(dim, dt=1.0, var=1.0, block_size=1, order_by_dim=true) do

    if !(dim in [2, 3, 4]) do
      raise %ArgumentError{message: "dim must be between 2 and 4"}
    end

    qQ =
      case dim do
        2 ->
          [[0.25*dt**4, 0.5*dt**3],
           [0.5*dt**3,      dt**2]]
        3 ->
          [[0.25*dt**4, 0.5*dt**3, 0.5*dt**2],
           [0.5*dt**3,      dt**2,     dt],
           [0.5*dt**2,       dt,        1]]
        _other ->
          [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
           [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
           [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
           [(dt**3)/6,  (dt**2)/2 ,  dt,        1.0]]
    end

    if order_by_dim do
      # block_diag(*[qQ]*block_size) * var
      # wtf python...
    else
      order_by_derivative(qQ, dim, block_size) * var
    end
  end

  @doc """
  Creates a block diagonal matrix from a list of matrices using Nx.

  ## Examples

  iex> Nx.tensor([[1, 2, 3], [4, 5, 6]], type: :f32) |> FilterEx.Utils.ravel()
  Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], type: :f32)


  """
  def block_diag(matrices) when is_list(matrices) do
    # Calculate the total size of the resulting matrix
    {total_rows, total_cols} =
      Enum.reduce(matrices, {0, 0}, fn mat, {rows, cols} ->
        {rows + tuple_size(Nx.shape(mat)) |> elem(0), cols + tuple_size(Nx.shape(mat)) |> elem(1)}
      end)

    # Initialize an empty matrix of zeros
    result = Nx.broadcast(0, {total_rows, total_cols})

    # Fill in each block along the diagonal
    Enum.reduce(matrices, {result, 0, 0}, fn mat, {acc, start_row, start_col} ->
      {rows, cols} = Nx.shape(mat)
      updated_matrix =
        acc
        |> Nx.slice_along_axis(start_row, rows, axis: 0)
        |> Nx.slice_along_axis(start_col, cols, axis: 1)
        |> Nx.add(mat)

      {updated_matrix, start_row + rows, start_col + cols}
    end)
    |> elem(0)
  end

  @doc """
  Flattens the input tensor into a 1D tensor using Nx.
  """
  def ravel(tensor) do
    # Get the total number of elements in the tensor
    num_elements = Nx.size(tensor)

    # Reshape the tensor into a 1D tensor
    Nx.reshape(tensor, {num_elements})
  end


  @deprecated
  @doc """
    Given a matrix Q, ordered assuming state space
        [x y z x' y' z' x'' y'' z''...]

    return a reordered matrix assuming an ordering of
       [ x x' x'' y y' y'' z z' y'']

    This works for any covariance matrix or state transition function

    Parameters
    ----------
    Q : np.array, square
        The matrix to reorder

    dim : int >= 1

       number of independent state variables. 3 for x, y, z

    block_size : int >= 0
        Size of derivatives. Second derivative would be a block size of 3
        (x, x', x'')


  """
  def order_by_derivative(qQ, dim, block_size) do
    nN = dim * block_size
    dD = zeros({nN, nN})
    # qQ = array(qQ)

    for {x, i} <- Enum.with_index(qQ |> ravel()), reduce: dD do
      dD ->
        f = Nx.eye(block_size) * x
        ix = div(i, dim) * block_size
        iy = rem(i, dim) * block_size
        # dD[ix .. (ix + block_size)][iy .. (iy + block_size)] = f
        dD |> Nx.put_slice({ix, iy}, f)
    end
  end
end
