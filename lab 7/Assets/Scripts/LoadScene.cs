using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneChanger : MonoBehaviour
{
    // Define the name of the scene you want to load
    public string sceneA;
    public string sceneB;

    // Track the current active scene
    private string currentScene;

    void Start()
    {
        // Set the initial active scene to sceneA
        if (SceneManager.GetActiveScene().name == sceneA)
        {
            currentScene = sceneA;
        }
        else
        {
            currentScene = sceneB;
        }
    }

    void Update()
    {
        // Check if the space bar is pressed
        if (Input.GetKeyDown(KeyCode.C))
        {
            // Toggle between scenes
            ToggleScene();
        }
    }

    private void ToggleScene()
    {
        // Check the current active scene and load the other one
    if (SceneManager.GetActiveScene().name == sceneA)
    {
        SceneManager.LoadScene(sceneB);
        currentScene = sceneB;
    }
    else
    {
        SceneManager.LoadScene(sceneA);
        currentScene = sceneA;
    }
    }
}
