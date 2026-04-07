# LiDAR_point_cloud_segmentation
Проект обработки LiDAR point cloud. Включает:
* сегментацию поверхности
* выделение границ дороги
* поиск дорожной разметки
* визуализацию (2D / 3D, изображение и GIF)

## Использование
1) Установка зависимостей

    ```pip install -r requirements.txt```

2) Команда для запуска

    ```python main.py --input_folder <путь к папке с .bin файлами > --type_of_vizualization 2D_gif```

## Аргументы

- --input_folder — путь к папке с .bin 

- --type_of_vizualization — тип визуализации 

Варианты: {2D_gif, 3D_gif, 2D_paint, 3D_paint}

- --ground_segmentation {RANSAC, grid_RANSAC, DBSCAN, Neighbours_grid_filter, iterative_filter, SalsaNext, SegFormer, combined} - способ сегментации плоскости (по умолчанию RANSAC)

- --boundaries_segmentation {extract_curb, extract_border, false} - способ сегментации границ (по умолчанию extract_curb)

- --markings_segmentation {true/false} - сегментация дорожной разметки (по умолчанию true)

- --gif_frames` — количество кадров (по умолчанию 1)
