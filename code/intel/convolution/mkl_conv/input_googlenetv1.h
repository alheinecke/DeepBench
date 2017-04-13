/*******************************************************************************
* Copyright 2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <vector>

struct conv_problem {
    int groups;
    int minibatch;
    int w;
    int h;
    int ic;
    int oc;
    int fw;
    int fh;
    int stride;
    int padd;
    int iters;
    const char *name;
};

static const std::vector<conv_problem> conv_problems = {
    {1, 32, 224, 224, 3, 64, 7, 7, 2, 3, 1000, "googlenet_v1:conv1/7x7_s2"},
    {1, 32, 56, 56, 64, 64, 1, 1, 1, 0, 1000, "googlenet_v1:conv2/3x3_reduce"},
    {1, 32, 56, 56, 64, 192, 3, 3, 1, 1, 1000, "googlenet_v1:conv2/3x3"},
    {1, 32, 28, 28, 192, 64, 1, 1, 1, 0, 1000, "googlenet_v1:inception_3a/1x1"},
    {1, 32, 28, 28, 192, 96, 1, 1, 1, 0, 1000, "googlenet_v1:inception_3a/3x3_reduce"},
    {1, 32, 28, 28, 96, 128, 3, 3, 1, 1, 1000, "googlenet_v1:inception_3a/3x3"},
    {1, 32, 28, 28, 192, 16, 1, 1, 1, 0, 1000, "googlenet_v1:inception_3a/5x5_reduce"},
    {1, 32, 28, 28, 16, 32, 5, 5, 1, 2, 1000, "googlenet_v1:inception_3a/5x5"},
    {1, 32, 28, 28, 192, 32, 1, 1, 1, 0, 1000, "googlenet_v1:inception_3a/pool_proj"},
    {1, 32, 28, 28, 256, 128, 1, 1, 1, 0, 1000, "googlenet_v1:inception_3b/1x1"},
    {1, 32, 28, 28, 256, 128, 1, 1, 1, 0, 1000, "googlenet_v1:inception_3b/3x3_reduce"},
    {1, 32, 28, 28, 128, 192, 3, 3, 1, 1, 1000, "googlenet_v1:inception_3b/3x3"},
    {1, 32, 28, 28, 256, 32, 1, 1, 1, 0, 1000, "googlenet_v1:inception_3b/5x5_reduce"},
    {1, 32, 28, 28, 32, 96, 5, 5, 1, 2, 1000, "googlenet_v1:inception_3b/5x5"},
    {1, 32, 28, 28, 256, 64, 1, 1, 1, 0, 1000, "googlenet_v1:inception_3b/pool_proj"},
    {1, 32, 14, 14, 480, 192, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4a/1x1"},
    {1, 32, 14, 14, 480, 96, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4a/3x3_reduce"},
    {1, 32, 14, 14, 96, 208, 3, 3, 1, 1, 1000, "googlenet_v1:inception_4a/3x3"},
    {1, 32, 14, 14, 480, 16, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4a/5x5_reduce"},
    {1, 32, 14, 14, 16, 48, 5, 5, 1, 2, 1000, "googlenet_v1:inception_4a/5x5"},
    {1, 32, 14, 14, 480, 64, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4a/pool_proj"},
    {1, 32, 4, 4, 512, 128, 1, 1, 1, 0, 1000, "googlenet_v1:loss1/conv"},
    {1, 32, 14, 14, 512, 160, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4b/1x1"},
    {1, 32, 14, 14, 512, 112, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4b/3x3_reduce"},
    {1, 32, 14, 14, 112, 224, 3, 3, 1, 1, 1000, "googlenet_v1:inception_4b/3x3"},
    {1, 32, 14, 14, 512, 24, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4b/5x5_reduce"},
    {1, 32, 14, 14, 24, 64, 5, 5, 1, 2, 1000, "googlenet_v1:inception_4b/5x5"},
    {1, 32, 14, 14, 512, 64, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4b/pool_proj"},
    {1, 32, 14, 14, 512, 128, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4c/1x1"},
    {1, 32, 14, 14, 512, 128, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4c/3x3_reduce"},
    {1, 32, 14, 14, 128, 256, 3, 3, 1, 1, 1000, "googlenet_v1:inception_4c/3x3"},
    {1, 32, 14, 14, 512, 24, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4c/5x5_reduce"},
    {1, 32, 14, 14, 24, 64, 5, 5, 1, 2, 1000, "googlenet_v1:inception_4c/5x5"},
    {1, 32, 14, 14, 512, 64, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4c/pool_proj"},
    {1, 32, 14, 14, 512, 112, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4d/1x1"},
    {1, 32, 14, 14, 512, 144, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4d/3x3_reduce"},
    {1, 32, 14, 14, 144, 288, 3, 3, 1, 1, 1000, "googlenet_v1:inception_4d/3x3"},
    {1, 32, 14, 14, 512, 32, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4d/5x5_reduce"},
    {1, 32, 14, 14, 32, 64, 5, 5, 1, 2, 1000, "googlenet_v1:inception_4d/5x5"},
    {1, 32, 14, 14, 512, 64, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4d/pool_proj"},
    {1, 32, 4, 4, 528, 128, 1, 1, 1, 0, 1000, "googlenet_v1:loss2/conv"},
    {1, 32, 14, 14, 528, 256, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4e/1x1"},
    {1, 32, 14, 14, 528, 160, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4e/3x3_reduce"},
    {1, 32, 14, 14, 160, 320, 3, 3, 1, 1, 1000, "googlenet_v1:inception_4e/3x3"},
    {1, 32, 14, 14, 528, 32, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4e/5x5_reduce"},
    {1, 32, 14, 14, 32, 128, 5, 5, 1, 2, 1000, "googlenet_v1:inception_4e/5x5"},
    {1, 32, 14, 14, 528, 128, 1, 1, 1, 0, 1000, "googlenet_v1:inception_4e/pool_proj"},
    {1, 32, 7, 7, 832, 256, 1, 1, 1, 0, 1000, "googlenet_v1:inception_5a/1x1"},
    {1, 32, 7, 7, 832, 160, 1, 1, 1, 0, 1000, "googlenet_v1:inception_5a/3x3_reduce"},
    {1, 32, 7, 7, 160, 320, 3, 3, 1, 1, 1000, "googlenet_v1:inception_5a/3x3"},
    {1, 32, 7, 7, 832, 32, 1, 1, 1, 0, 1000, "googlenet_v1:inception_5a/5x5_reduce"},
    {1, 32, 7, 7, 32, 128, 5, 5, 1, 2, 1000, "googlenet_v1:inception_5a/5x5"},
    {1, 32, 7, 7, 832, 128, 1, 1, 1, 0, 1000, "googlenet_v1:inception_5a/pool_proj"},
    {1, 32, 7, 7, 832, 384, 1, 1, 1, 0, 1000, "googlenet_v1:inception_5b/1x1"},
    {1, 32, 7, 7, 832, 192, 1, 1, 1, 0, 1000, "googlenet_v1:inception_5b/3x3_reduce"},
    {1, 32, 7, 7, 192, 384, 3, 3, 1, 1, 1000, "googlenet_v1:inception_5b/3x3"},
    {1, 32, 7, 7, 832, 48, 1, 1, 1, 0, 1000, "googlenet_v1:inception_5b/5x5_reduce"},
    {1, 32, 7, 7, 48, 128, 5, 5, 1, 2, 1000, "googlenet_v1:inception_5b/5x5"},
    {1, 32, 7, 7, 832, 128, 1, 1, 1, 0, 1000, "googlenet_v1:inception_5b/pool_proj"},
};
