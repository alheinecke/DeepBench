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
    {1, 128, 224, 224, 3, 64, 3, 3, 1, 1, 1000, ""},
    {1, 128, 112, 112, 64, 128, 3, 3, 1, 1, 1000, ""},
    {1, 128, 56, 56, 128, 256, 3, 3, 1, 1, 1000, ""},
    {1, 128, 56, 56, 256, 256, 3, 3, 1, 1, 1000, ""},
    {1, 128, 28, 28, 256, 512, 3, 3, 1, 1, 1000, ""},
    {1, 128, 28, 28, 512, 512, 3, 3, 1, 1, 1000, ""},
    {1, 128, 14, 14, 512, 512, 3, 3, 1, 1, 1000, ""},
};
