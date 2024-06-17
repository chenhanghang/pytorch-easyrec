# -*- ecoding: utf-8 -*-
from . import *
from .rank.lr import *
from .rank.fm import *
from .rank.nfm import *
from .rank.afm import *
from .rank.fnn import * # 深度，Embedding + MLP 范式
from .rank.deep_crossing import * # 引入残差
from .rank.wide_deep import *
from .rank.deep_fm import * # 深度和FM结合
from .rank.dcn import *
from .rank.pnn import *

from .srank.din import *