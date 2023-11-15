using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class LoadScenesA : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("LoadSceneA");
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C)) {
            Debug.Log("C key was pressed.");
            LoadA("NatureScene");
        }
    }
   

    void LoadA(string scenename)
    {
        Debug.Log("sceneName to load: " + scenename);
        SceneManager.LoadScene(scenename);
    }
}
