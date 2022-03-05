def load_breast_cancer(*, return_X_y=False, as_frame=False):
    """Load and return the breast cancer wisconsin dataset (classification).
    The breast cancer dataset is a classic and very easy binary classification
    dataset.
    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============
    The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is
    downloaded from:
    https://goo.gl/U2Uwz2
    Read more in the :ref:`User Guide <breast_cancer_dataset>`.
    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
        .. versionadded:: 0.18
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.
        .. versionadded:: 0.23
    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (569, 30)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target : {ndarray, Series} of shape (569,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names : list
            The names of the dataset columns.
        target_names : list
            The names of target classes.
        frame : DataFrame of shape (569, 31)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
            .. versionadded:: 0.23
        DESCR : str
            The full description of the dataset.
        filename : str
            The path to the location of the data.
            .. versionadded:: 0.20
    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarrays by default. The first contains a 2D ndarray of
        shape (569, 30) with each row representing one sample and each column
        representing the features. The second ndarray of shape (569,) contains
        the target samples.  If `as_frame=True`, both arrays are pandas objects,
        i.e. `X` a dataframe and `y` a series.
        .. versionadded:: 0.18
    Examples
    --------
    Let's say you are interested in the samples 10, 50, and 85, and want to
    know their class name.
    >>> from sklearn.datasets import load_breast_cancer
    >>> data = load_breast_cancer()
    >>> data.target[[10, 50, 85]]
    array([0, 1, 0])
    >>> list(data.target_names)
    ['malignant', 'benign']
    """
    data_file_name = "breast_cancer.csv"
    data, target, target_names, fdescr = load_csv_data(
        data_file_name=data_file_name, descr_file_name="breast_cancer.rst"
    )

    feature_names = np.array(
        [
            "checkin_acc",
            "duration",
            "credit history",
            "purpose",
            "amount",
            "saving_acc",
            "present_emp_since",
            "inst_rate",
            "personal_status",
            "other_debtors",
            "residing_since",
            "property",
            "age",
            "inst-plans",
            "housing",
            "num_credits",
            "job",
            "dependents",
            "telephone",
            "foreign_worker",
            "status",
        ]
    )

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_breast_cancer", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
        filename=data_file_name,
        data_module=DATA_MODULE,
    )

