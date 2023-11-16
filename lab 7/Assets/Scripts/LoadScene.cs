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
        currentScene = sceneA;
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

    void ToggleScene()
    {
        // Determine the next scene to load based on the current active scene
        string nextScene;
        
        if(currentScene == sceneA)
            nextScene = sceneB;

        else if(currentScene == sceneB)
            nextScene = sceneA;
        else
            nextScene = sceneA; 

        // Load the next scene
        SceneManager.LoadScene(nextScene);

        // Update the current active scene
        currentScene = nextScene;
    }
}
