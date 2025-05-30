using Microsoft.AspNetCore.Mvc;
using RecommenderAPI.Models;
using RecommenderAPI.Services;

namespace RecommenderAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class RecommendationController : ControllerBase
    {
        private readonly RecommendationService _service;

        public RecommendationController(RecommendationService service)
        {
            _service = service;
        }

        [HttpPost]
        public async Task<IActionResult> Recommend([FromBody] StudentProfile profile)
        {
            var result = await _service.GetRecommendationsAsync(profile);
            return Ok(result);
        }
    }
}
