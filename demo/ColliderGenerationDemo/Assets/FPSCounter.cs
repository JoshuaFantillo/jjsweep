using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;


// From: https://stackoverflow.com/questions/75322179/is-this-a-correct-way-to-implement-fps-counter-in-unity
[RequireComponent(typeof(TMP_Text))]
public class FPSCounter : MonoBehaviour
{
    [SerializeField][Range(0f, 1f)] private float _expSmoothingFactor = 0.9f;
    [SerializeField] private float _refreshFrequency = 0.4f;
    [SerializeField] private TMP_Text _text;

    private float _timeSinceUpdate = 0f;
    private float _averageFps = 1f;

    private void Update()
    {
        // Exponentially weighted moving average (EWMA)
        _averageFps = _expSmoothingFactor * _averageFps + (1f - _expSmoothingFactor) * 1f / Time.unscaledDeltaTime;

        if (_timeSinceUpdate < _refreshFrequency)
        {
            _timeSinceUpdate += Time.deltaTime;
            return;
        }

        int fps = Mathf.RoundToInt(_averageFps);
        _text.text = fps.ToString();

        _timeSinceUpdate = 0f;
    }
}
