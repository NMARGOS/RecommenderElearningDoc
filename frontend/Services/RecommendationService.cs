using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using RecommenderAPI.Models;

namespace RecommenderAPI.Services
{
    public class RecommendationService
    {
        private readonly HttpClient _httpClient;

        public RecommendationService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<List<RecommendationResults>> GetRecommendationsAsync(StudentProfile profile)
        {
            var response = await _httpClient.PostAsJsonAsync("http://localhost:5000/recommend", profile);
            response.EnsureSuccessStatusCode(); // Throws if status is not 2xx
            return await response.Content.ReadFromJsonAsync<List<RecommendationResults>>();
        }
    }
}
