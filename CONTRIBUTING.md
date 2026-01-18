# Contributing to GFN | Manifold

Thank you for your interest in contributing to Geodesic Flow Networks (Manifold)!

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, PyTorch version, GPU)

### Suggesting Enhancements

For feature requests:
- Describe the feature and its motivation
- Provide use cases
- Consider backward compatibility

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Include unit tests for new functionality
   - Update documentation if needed

4. **Test your changes**
   ```bash
   # Run unit tests
   python -m pytest tests/unit/
   
   # Run integration tests
   python tests/test_integration.py
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: brief description"
   ```

6. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Keep functions focused and concise
- Add comments for complex logic

## Documentation

- Update relevant `.md` files
- Add docstrings (Google style)
- Include examples for new features

## Testing

- Write unit tests for new code
- Ensure all tests pass before submitting PR
- Aim for high test coverage

## Questions?

Feel free to open an issue for any questions!
