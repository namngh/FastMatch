# FastMatch

Base on C++ implementation [subokita/FAsT-Match](https://github.com/subokita/FAsT-Match) and Mathlab implementation [here](https://www.eng.tau.ac.il/~simonk/FastMatch/)

## Installation

```
git+https://github.com/namngh/FastMatch@main
```

## Usage

```python
from cv2
from fast_match import fast_match

image = cv2.imread("image.png")
template = cv2.imread("template.png")

corners = fast_match.FastMatch(epsilon=0.15, delta=0.85, photometric_invariance=False, min_scale=0.5, max_scale=2).run(image, template)
```

## Contributing

Pull requests are welcome. For major changes,
please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)