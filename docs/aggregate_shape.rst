Snapshot Segmentation
=====================

Segment phase snapshots of elongating organoids using the ``analyze_aggregate_shape.py`` script:

.. code-block:: bash

    ./analyze_aggregate_shape.py path/to/data \
          --processes 1 \
          --dark-aggregates \
          --image-norm 1.5 \
          --threshold 0.05 \
          --border-pixels '-1' \
          --min-mask-size 2000

See :py:func:`~organoid_shape_tools.aggregate_shape.analyze_aggregate_image` for details about the segmentation algorithm.

After segmenting several images, use the ``merge_analyze_aggregate_shape.py`` script:

.. code-block:: bash

    ./merge_analyze_aggregate_shape.py path/to/data

Which will combine all the files under ``path/to/data`` into a single merged spreadsheet.

See :py:func:`~organoid_shape_tools.aggregate_shape.extract_image_stats` for details about the statistics calculated.

If analyzing elongating organoids, use the ``plot_analyze_aggregate_shape.py`` script:

.. code-block:: bash

    ./plot_analyze_aggregate_shape.py path/to/data

Which will plot the elongation timelines for each aggregate type in the experiment.

If analyzing the TBXT KD organoids, use the ``plot_analyze_aggregate_hands.py`` script:

.. code-block:: bash

  ./plot_analyze_aggregate_hands.py path/to/data

Which will plot the convexity analysis timelines for each aggregate type in the experiment.

See :py:func:`~organoid_shape_tools.aggregate_shape.load_convexity_contours` for details about the convexity analysis.

Segmentation API
================

.. automodule:: organoid_shape_tools.aggregate_shape
  :members:
