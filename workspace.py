import openai
import pkg_resources

def check_openai_version():
    """Check OpenAI version and configuration details"""
    
    # Get version
    try:
        version = pkg_resources.get_distribution("openai").version
        print(f"OpenAI package version: {version}")
    except pkg_resources.DistributionNotFound:
        print("OpenAI package not found")
        return
    
    # Check if it's the old or new client
    print(f"OpenAI module version attribute: {getattr(openai, '__version__', 'Not found')}")
    
    # Check available attributes to determine API style
    print("\nAvailable OpenAI attributes:")
    attrs = [attr for attr in dir(openai) if not attr.startswith('_')]
    for attr in sorted(attrs):
        print(f"  - {attr}")
    
    # Test client initialization patterns
    print("\nTesting client initialization:")
    
    # Test new client (v1.x)
    try:
        from openai import OpenAI
        print("✓ New OpenAI client import successful")
        
        # Try to create client (this might fail without API key, but shows if syntax works)
        try:
            client = OpenAI(api_key="test-key")
            print("✓ New client initialization syntax works")
        except Exception as e:
            if "api_key" in str(e).lower():
                print("✓ New client syntax works (just needs real API key)")
            else:
                print(f"✗ New client initialization error: {e}")
                
    except ImportError as e:
        print(f"✗ New OpenAI client import failed: {e}")
    
    # Test old client pattern
    try:
        import openai as old_openai
        if hasattr(old_openai, 'ChatCompletion'):
            print("✓ Old OpenAI client pattern available")
        else:
            print("✗ Old OpenAI client pattern not available")
    except Exception as e:
        print(f"✗ Old client test error: {e}")

if __name__ == "__main__":
    check_openai_version()