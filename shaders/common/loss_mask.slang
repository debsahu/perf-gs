// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_LOSS_MASK_SLANG
#define TRAINER_LOSS_MASK_SLANG

// LOSS_MASK = 1 when the run has per-image loss masks (--masks). The mask is the target texture's
// alpha (1 keep, 0 drop): a dropped pixel gives no color gradient and does not drive densification
// or pruning. Off by default, and then every shader compiles exactly as without this header.
#ifndef LOSS_MASK
#define LOSS_MASK 0
#endif

#endif // TRAINER_LOSS_MASK_SLANG
