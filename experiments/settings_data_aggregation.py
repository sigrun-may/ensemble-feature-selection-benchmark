# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import initialize_settings


def main():
    initialize_settings.init_aggregation()


if __name__ == "__main__":
    print("No data storage for aggregation")
    main()
