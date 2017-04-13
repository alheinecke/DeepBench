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
    {1, 256, 231, 231, 3, 96, 11, 11, 4, 0, 1000, "overfeat:conv1"},
    {1, 256, 28, 28, 96, 256, 5, 5, 1, 0, 1000, "overfeat:conv2"},
    {1, 256, 12, 12, 256, 512, 3, 3, 1, 1, 1000, "overfeat:conv3"},
    {1, 256, 12, 12, 512, 1024, 3, 3, 1, 1, 1000, "overfeat:conv4"},
    {1, 256, 12, 12, 1024, 1024, 3, 3, 1, 1, 1000, "overfeat:conv5"},
};
