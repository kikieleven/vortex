#ifndef COMMON_H
#define COMMON_H

#ifndef TYPE
#define TYPE float
#endif

#define LSTM_SEQ    4
#define LSTM_IN     8
#define LSTM_HIDDEN 8
#define LSTM_GATES  4   /* i, f, g, o in PyTorch's packed order */

#endif // COMMON_H
