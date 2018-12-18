===========
Basic Usage
===========

We have provided a script, `christmais_time.py` to easily generate your stylized Quick, Draw! images.
In order to use it, simply run the following command:

.. code-block:: shell

   python -m christmais.tasks.christmais_time     \
       --input=<Input string to draw from>        \
       --style=<Path to style image>              \
       --output=<Unique name of output file>      \
       --model-path=<Path to model.ckpt>          \
       --categories-path=<Path to categories.txt> \
       --webdriver-path=<Path to webdriver>


If you followed the setup instructions above, then the default values for the
paths should suffice, you only need to supply `--input`, `--style`, and
`--output`.


As an example, let's say I want to use the string `Thinking Machines` as our
basis with the style of *Ang Kiukok's*
`*Fishermen* <https://lifestyle.inquirer.net/263837/starting-bid-ang-kiukok-manansala-p12-million/>`_
(`ang_kiukok.jpg`), then, my command will look like this:

.. code-block:: shell

   python -m christmais.tasks.christmais_time \
       --input="Thinking Machines"            \
       --style=./path/to/ang_kiukok.png       \
       --output=tmds-output

This will then generate the output image in `./artifacts/`:

.. image:: https://storage.googleapis.com/tm-christmais/assets/tmds.png
   :height: 170
