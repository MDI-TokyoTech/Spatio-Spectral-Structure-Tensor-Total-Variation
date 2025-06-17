function normalized01_x = normalize01(x)
    x_max = max(x, [], 'all');
    x_min = min(x, [], 'all');

    normalized01_x = (x - x_min) / (x_max - x_min);
end