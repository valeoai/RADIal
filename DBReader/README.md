# DBReader
DBReader is a library to read individual sequence of the dataset.
It's composed of 2 main function:
* ASyncReader, which reads the data in the order there were recorded
* SyncReader, which reads in a synchronized mode all the data. A master sensor (usually the slowest one) has to be specified together with a time tolerance window.

Install it with
```
pip install .
```
Uninstall with 
```
pip uninstall db-reader
```

#### How to use:
Please, check the examples folder that shows several uses of the library
